/**
 * Boundary condition representing momentum-space transport across a boundary
 * between two separate grids. The upper flux boundary of the first grid must
 * be equal to the lower flux boundary of the second grid, but the resolution
 * in \xi may be different.
 *
 * This boundary condition can be applied to either of the momentum space grids.
 *      ________________ ________________
 *  ^  |                x                |
 *  |  |     GRID 1     x     GRID 2     |
 * xi  |________________x________________|
 *     p0            p ---->            pmax
 *
 * (the boundary condition is applied on the boundary marked with the crosses)
 */

#include "FVM/BlockMatrix.hpp"
#include "FVM/Equation/BoundaryConditions/PXiExternalKineticKinetic.hpp"


using namespace DREAM::FVM::BC;


/**
 * Constructor.
 */
PXiExternalKineticKinetic::PXiExternalKineticKinetic(
    DREAM::FVM::Grid *grid, DREAM::FVM::Grid *lowerGrid, DREAM::FVM::Grid *upperGrid,
    const DREAM::FVM::Operator *eqn, const len_t id_f_low, const len_t id_f_upp,
    enum condition_type ctype
) : PXiAdvectionDiffusionBoundaryCondition(grid, eqn),
    lowerGrid(lowerGrid), upperGrid(upperGrid),
    id_f_low(id_f_low), id_f_upp(id_f_upp), type(ctype) {
    
    SetName("PXiExternalKineticKinetic");
}

/**
 * Destructor.
 */
PXiExternalKineticKinetic::~PXiExternalKineticKinetic() { }


/**
 * Returns the number of non-zero elements set by this boundary
 * condition, per row, in the linear operator matrix.
 */
len_t PXiExternalKineticKinetic::GetNumberOfNonZerosPerRow() const {
    if (this->type == TYPE_DENSITY) {
        const len_t
            ln2 = this->lowerGrid->GetMomentumGrid(0)->GetNp2(),
            un2 = this->upperGrid->GetMomentumGrid(0)->GetNp2();
        return 2*(ln2 + 2*un2);
    } else
        // On kinetic grids, this term does not add any non-zero
        // elements which have not already been added by the regular
        // advection-diffusion terms.
        return 0;
}

/**
 * Returns the number of non-zero elements set by this boundary
 * condition, per row, in the jacobian matrix.
 */
len_t PXiExternalKineticKinetic::GetNumberOfNonZerosPerRow_jac() const {
    // XXX here we assume that all momentum grids are the same
    // at all radii
    const len_t
        ln2 = this->lowerGrid->GetMomentumGrid(0)->GetNp2(),
        un2 = this->upperGrid->GetMomentumGrid(0)->GetNp2();

    /*AdvectionDiffusionTerm *adt = oprtr->GetAdvectionDiffusion();
    len_t nnzOffDiag = adt->GetNumberOfNonZerosPerRow_jac() - adt->GetNumberOfNonZerosPerRow();*/
    len_t nnzOffDiag = 0;
    switch (this->type) {
        case TYPE_LOWER:
            return (un2 > ln2 ? 3 : 2) + nnzOffDiag;
        case TYPE_UPPER:
            return (ln2 > un2 ? 3 : 2) + nnzOffDiag;

        case TYPE_DENSITY:
            return 2*(ln2 + 2*un2);

        default: return 0;
    }
}

/**
 * Construct advection/diffusion coefficients to use on the upper grid.
 * These are constructed by averaging the coefficients on the lower grid
 * over the range in \xi covered by each grid cell on the upper grid.
 */
bool PXiExternalKineticKinetic::Rebuild(const real_t, UnknownQuantityHandler *uqh) {
    this->fLow = uqh->GetUnknownData(id_f_low);
    this->fUpp = uqh->GetUnknownData(id_f_upp);
    
    return true;
}

/**
 * Add flux to jacobian block.
 */
bool PXiExternalKineticKinetic::AddToJacobianBlock(
    const len_t uqtyId, const len_t derivId, Matrix *jac, const real_t *x
) {
    bool contributes = (derivId == uqtyId);
    if (derivId == uqtyId)
        // Note that this will also set the off-diagonal block 
        // corresponding to the quantity/ies that is/are not
        // 'uqtyId'.
        this->AddToMatrixElements(jac, nullptr);

    // Handle derivatives of coefficients (we assume that the coefficients
    // do not depend on either of the distribution functions...)
    if (derivId != this->id_f_low && derivId != this->id_f_upp)
        contributes |=
            this->PXiAdvectionDiffusionBoundaryCondition::AddPartialJacobianContributions(
                uqtyId, derivId, jac, x, false
            );

    return contributes;
}

/**
 * Add flux to linearized operator matrix.
 * 
 * mat: Matrix to add boundary conditions to.
 * rhs: Right-hand-side vector (not used).
 */
void PXiExternalKineticKinetic::AddToMatrixElements(
    Matrix *mat, real_t*
) {
    // Reset matrix offsets...
    const len_t rowOffset = mat->GetRowOffset();
    const len_t colOffset = mat->GetColOffset();
    mat->SetOffset(rowOffset, 0);

    // Get matrix offsets of f_low and f_upp
    BlockMatrix *bm = static_cast<BlockMatrix*>(mat);
    len_t lowOffset = bm->GetOffsetById(id_f_low);
    len_t uppOffset = bm->GetOffsetById(id_f_upp);

    // Only the matrix rows owned by this rank are set
    PetscInt rstart, nloc;
    mat->GetOwnershipRange(&rstart, &nloc);
    auto isLocal = [rowOffset,rstart,nloc](const len_t I) {
        const PetscInt row = (PetscInt)rowOffset + (PetscInt)I;
        return (row >= rstart && row < rstart+nloc);
    };

    __SetElements(
        [mat,lowOffset,isLocal](const len_t I, const len_t J, const real_t V) { if (isLocal(I)) mat->SetElement(I, lowOffset+J, V); },
        [mat,uppOffset,isLocal](const len_t I, const len_t J, const real_t V) { if (isLocal(I)) mat->SetElement(I, uppOffset+J, V); }
    );

    // Restore matrix offset
    mat->SetOffset(rowOffset, colOffset);
}

/**
 * Add flux to function vector.
 */
void PXiExternalKineticKinetic::AddToVectorElements_c(
    real_t *vec, const real_t*,
    const real_t *const*, const real_t *const* df1, const real_t *const*,
    const real_t *const*, const real_t *const* dd11, const real_t *const* dd12,
    const real_t *const*, const real_t *const*,
    jacobian_interp_mode
) {
    const real_t *fLow = this->fLow;
    const real_t *fUpp = this->fUpp;

    __SetElements(
        [vec,fLow](const len_t I, const len_t J, const real_t V) { vec[I] += V*fLow[J]; },
        [vec,fUpp](const len_t I, const len_t J, const real_t V) { vec[I] += V*fUpp[J]; },
        df1, dd11, dd12
    );
}

/**
 * Internal routine for setting matrix/vector elements.
 */
void PXiExternalKineticKinetic::__SetElements(
    std::function<void(const len_t, const len_t, const real_t)> fLow,
    std::function<void(const len_t, const len_t, const real_t)> fUpp
) {
    // The coefficients of the lower grid on its upper p boundary, collected
    // on every rank by the advection-diffusion term (each rank only stores
    // the coefficients of the cells it owns)
    const AdvectionDiffusionTerm *adt = oprtr->GetAdvectionDiffusion();
    boundaryCoefficient Ap, Dpp;
    if (adt != nullptr) {
        Ap  = [adt](const len_t ir, const len_t j) { return adt->GetUpperBoundaryF1(ir, j); };
        Dpp = [adt](const len_t ir, const len_t j) { return adt->GetUpperBoundaryD11(ir, j); };
    }

    this->__SetElements(fLow, fUpp, Ap, Dpp);
}

/**
 * Same as above, with the coefficients (on all faces of the lower grid)
 * given explicitly.
 */
void PXiExternalKineticKinetic::__SetElements(
    std::function<void(const len_t, const len_t, const real_t)> fLow,
    std::function<void(const len_t, const len_t, const real_t)> fUpp,
    const real_t *const* cAp, const real_t *const* cDpp,
    const real_t *const* /*cDpx*/
) {
    // Coefficient on the last p1 face of xi line j at radius ir
    auto onBoundary = [this](const real_t *const* c) -> boundaryCoefficient {
        if (c == nullptr)
            return boundaryCoefficient();
        return [this,c](const len_t ir, const len_t j) {
            const len_t lnp = this->lowerGrid->GetMomentumGrid(ir)->GetNp1();
            return c[ir][j*(lnp+1)+lnp];
        };
    };

    this->__SetElements(fLow, fUpp, onBoundary(cAp), onBoundary(cDpp));
}

void PXiExternalKineticKinetic::__SetElements(
    std::function<void(const len_t, const len_t, const real_t)> fLow,
    std::function<void(const len_t, const len_t, const real_t)> fUpp,
    boundaryCoefficient cAp, boundaryCoefficient cDpp
) {
    const len_t nr = this->grid->GetNr();
    len_t loffset = 0, uoffset = 0;

    for (len_t ir = 0; ir < nr; ir++) {
        const MomentumGrid *lmg = this->lowerGrid->GetMomentumGrid(ir);
        const MomentumGrid *umg = this->upperGrid->GetMomentumGrid(ir);

        const len_t
            lnp = lmg->GetNp1(), lnxi = lmg->GetNp2(),
            unp = umg->GetNp1(), unxi = umg->GetNp2();

        const real_t
            *lp    = lmg->GetP1(),
            *up    = umg->GetP1(),
            *lxi   = lmg->GetP2(),
            *uxi   = umg->GetP2(),
            *lxi_f = lmg->GetP2_f(),
            *uxi_f = umg->GetP2_f(),
            *ldp   = lmg->GetDp1(),
            *udp   = umg->GetDp1(),
            *ldxi  = lmg->GetDp2(),
            *udxi  = umg->GetDp2();

        const bool hasAp = (bool)cAp, hasDpp = (bool)cDpp;

        const real_t
            *lVp   = this->lowerGrid->GetVp(ir),
            *lVp_f = this->lowerGrid->GetVp_f1(ir),
            *uVp   = this->upperGrid->GetVp(ir),
            *VpVol = this->grid->GetVpVol();

        // j = xi index on lower grid
        // J = xi index on upper grid
        for (len_t j = 0, Jj = 0; j < lnxi; j++) {
            // Coefficients on the upper p boundary of this xi line
            const real_t Ap  = hasAp  ? cAp(ir, j)  : 0;
            const real_t Dpp = hasDpp ? cDpp(ir, j) : 0;

            // Locate correct J (first J such that uxi[J] >= lxi[j])...
            while (uxi[Jj] < lxi[j] && Jj < unxi-1)
                Jj++;

            for (len_t J = 0; J < unxi; J++) {
                real_t dxiBar = std::min(lxi_f[j+1], uxi_f[J+1]) - std::max(lxi_f[j], uxi_f[J]);

                if (dxiBar <= 0)
                    continue;

                // Shortcuts to indices...
                len_t
                    lidx   = j*lnp+lnp-1,
                    lidx_f = j*(lnp+1)+lnp,
                    uidx   = Jj*unp,
                    uidx_m = (Jj-1)*unp,
                    fidx;

                // Set indices for f(r,p,xi) and FVM geometric factor
                // based on which quantity we're building the flux
                // for (for f_hot, f_RE or n_RE)
                real_t Vd;
                if (this->type == TYPE_LOWER) {
                    // if(!lVp[lidx])
                    if(lowerGrid->IsNegativePitchTrappedIgnorableCell(ir,j))
                        continue;
                    fidx = loffset + lidx;
                    Vd   = lVp_f[lidx_f] / (lVp[lidx] * ldp[lnp-1]);
                } else if (this->type == TYPE_UPPER) {
                    len_t J_tmp = J;
                    // if negative pitch trapped runaway, find the
                    // cell containing -uxi to which we add the flux instead
                    if(upperGrid->IsNegativePitchTrappedIgnorableCell(ir,J))
                        while(uxi_f[J_tmp+1]<-uxi[J] && J_tmp<unxi)
                            J_tmp++;

                    fidx = uoffset + J_tmp*unp;
                    real_t fac = dxiBar / udxi[J_tmp];
                    Vd   = -fac*lVp_f[lidx_f] / (uVp[J_tmp*unp] * udp[0]);
                } else /*if (this->type == TYPE_DENSITY)*/ {
                    fidx = ir;
                    Vd   = -lVp_f[lidx_f] * ldxi[j] / VpVol[ir];
                }

                // Interpolation coefficients...
                real_t delta1, delta2;
                if (hasAp)
                    delta2 = Ap > 0 ? 1.0 : 0.0;

                if (Jj == 0)
                    delta1 = 1.0;
                else if (uxi[Jj] < lxi[j])
                    delta1 = 0.0;
                else
                    delta1 = (lxi[j]-uxi[Jj-1]) / (uxi[Jj]-uxi[Jj-1]);

                //////////////////////
                // Advection
                if (hasAp)
                    fLow(fidx, loffset+lidx, delta2*Ap*Vd);

                if (delta1 != 0 && hasAp)
                    fUpp(fidx, uoffset+uidx, (1-delta2)*delta1*Ap*Vd);
                if (delta1 != 1 && hasAp)
                    fUpp(fidx, uoffset+uidx_m, (1-delta2)*(1-delta1)*Ap*Vd);

                //////////////////////
                // P-P diffusion
                real_t dp = up[0]-lp[lnp-1];
                if (hasDpp)
                    fLow(fidx, loffset+lidx, Dpp*Vd/dp);

                if (delta1 != 0 && hasDpp)
                    fUpp(fidx, uoffset+uidx, -delta1*Dpp*Vd/dp);
                if (delta1 != 1 && hasDpp)
                    fUpp(fidx, uoffset+uidx_m, -(1-delta1)*Dpp*Vd/dp);

                //////////////////////
                // P-XI diffusion (TODO)

                if (this->type == TYPE_LOWER || this->type == TYPE_DENSITY)
                    break;
            }
        }

        loffset += lnp*lnxi;
        uoffset += unp*unxi;
    }
}
