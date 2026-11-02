/**
 * Implementation of a diffusive transport term which can be applied to both
 * kinetic and fluid grids, and which allows one to prescribe the diffusion
 * coefficient in time and phase space.
 */

#include <algorithm>
#include <type_traits>
#include "DREAM/Equations/TransportPrescribed.hpp"
#include "FVM/Interpolator1D.hpp"
#include "FVM/Interpolator3D.hpp"


/**
 * Constructor.
 */
template<typename T>
DREAM::TransportPrescribed<T>::TransportPrescribed(
    DREAM::FVM::Grid *grid,
    const len_t nt, const len_t nr, const len_t np1, const len_t np2,
    const real_t **coeff, const real_t *t, const real_t *r,
    const real_t *p1, const real_t *p2, enum DREAM::FVM::Interpolator3D::momentumgrid_type inptype,
    enum DREAM::FVM::Interpolator3D::momentumgrid_type gridtype,
    enum DREAM::FVM::Interpolator3D::interp_method interpmethod,
    bool allocCoefficients
) : T(grid, allocCoefficients),
    nt(nt), nr(nr), np1(np1), np2(np2),
    coeff(coeff), t(t), r(r), p1(p1), p2(p2),
    momtype(inptype), gridtype(gridtype), interpmethod(interpmethod) {

    this->T::SetName("TransportPrescribed");
    
    // Interpolate input coefficient onto 'grid'...
    InterpolateCoefficient();
}

/**
 * Destructor.
 */
namespace DREAM {
template<typename T>
TransportPrescribed<T>::~TransportPrescribed() {
    if (this->prescribedCoeff != nullptr)
        delete this->prescribedCoeff;
    if (this->interpolateddata != nullptr) {
        delete [] this->interpolateddata;
    }

    delete [] this->coeff[0];
    delete [] this->coeff;
    delete [] this->t;
    delete [] this->r;
    delete [] this->p1;
    delete [] this->p2;
}
}


/**
 * Function called whenever the computational grid has been
 * rebuilt. Here, we will need to interpolate the prescribed
 * diffusion coefficient onto the new grid.
 */
template<typename T>
bool DREAM::TransportPrescribed<T>::GridRebuilt() {
    //  Call GridRebuilt() on base class...
    this->T::GridRebuilt();

    // Interpolate prescribed coefficient onto new
    // phase space grid...
    InterpolateCoefficient();

    return true;
}

/**
 * Interpolate the input coefficient onto the phase space
 * grid used by this EquationTerm.
 */
template<typename T>
void DREAM::TransportPrescribed<T>::InterpolateCoefficient() {
    real_t **newdata = new real_t*[nt];
    // Drr is defined on the radial flux grid so we
    // XXX assume that all momentum grids are the same
    // and one momentum grid's worth of cells to the total
    // number of cells...
    const len_t N  =
        this->grid->GetNCells() + this->grid->GetMomentumGrid(0)->GetNCells();

    newdata[0] = new real_t[nt*N];
    real_t* newt = new real_t[nt];

    for (len_t i = 0; i < nt; i++) {
        if (i > 0)
            newdata[i] = newdata[i-1] + N;
        newt[i] = t[i];

        DREAM::FVM::Interpolator3D intp3(
            nr, np2, np1, r, p2, p1, coeff[i],
            momtype, interpmethod, false
        );
        intp3.Eval(this->grid, this->gridtype, FVM::FLUXGRIDTYPE_RADIAL, newdata[i]);
    }

    if (this->prescribedCoeff != nullptr) {
        delete this->prescribedCoeff;
        delete [] this->interpolateddata;
    }

    this->prescribedCoeff = new DREAM::FVM::Interpolator1D(
        nt, N, newt, newdata[0]
    );

    // This data is now used by 'prescribedCoeff', but we
    // need to keep a pointer in this class so that we can
    // clean it up later...
    this->interpolateddata = newdata;
}

/**
 * Rebuild this term by evaluating and setting the diffusion
 * coefficient for the next time step.
 *
 * Only the radial faces of the cells whose matrix rows are owned by
 * this rank are set (see 'EquationTerm::GetLocalRange()'), and the
 * coefficient is interpolated in time on those faces only, linearly
 * as 'prescribedCoeff->Eval()' does on the whole grid.
 */
template<typename T>
void DREAM::TransportPrescribed<T>::Rebuild(
    const real_t tv, const real_t, DREAM::FVM::UnknownQuantityHandler*
) {
    const len_t nr = this->grid->GetNr();
    // XXX here we assume that all momentum grids are the same...
    const len_t N   = this->grid->GetMomentumGrid(0)->GetNCells();
    const len_t np1 = this->grid->GetMomentumGrid(0)->GetNp1();

    // Time points bracketing 'tv' (same search as Interpolator1D, which
    // also extrapolates linearly outside [t[0], t[nt-1]])
    const real_t *y1 = this->interpolateddata[0], *y2 = y1;
    real_t w = 0;
    if (this->nt > 1) {
        const bool increasing = (this->t[1] > this->t[0]);
        len_t a = 0, b = this->nt-1;
        while (b - a > 1) {
            const len_t c = (a+b)/2;
            if ((this->t[c] >= tv) == increasing) b = c;
            else a = c;
        }
        y1 = this->interpolateddata[a];
        y2 = this->interpolateddata[a+1];
        w  = (tv - this->t[a]) / (this->t[a+1] - this->t[a]);
    }

    // Iterate over the radial flux grid...
    for (len_t ir = 0; ir < nr+1; ir++) {
        // Radial face 'ir' is the upper face of cell ir-1 and the lower
        // face of cell ir: set it where this rank owns either cell
        len_t i1lo[2], i1hi[2], i2lo[2], i2hi[2];
        const bool has[2] = {
            ir > 0  && this->GetLocalRange(ir-1, i1lo[0], i1hi[0], i2lo[0], i2hi[0]),
            ir < nr && this->GetLocalRange(ir,   i1lo[1], i1hi[1], i2lo[1], i2hi[1])
        };
        if (!has[0] && !has[1])
            continue;

        auto isLocalFace = [&](const len_t i, const len_t j) {
            for (len_t k = 0; k < 2; k++)
                if (has[k] && i >= i1lo[k] && i < i1hi[k] && j >= i2lo[k] && j < i2hi[k])
                    return true;
            return false;
        };

        // Bounding box of the faces to set
        const len_t
            ilo = has[0] && has[1] ? std::min(i1lo[0], i1lo[1]) : (has[0] ? i1lo[0] : i1lo[1]),
            ihi = has[0] && has[1] ? std::max(i1hi[0], i1hi[1]) : (has[0] ? i1hi[0] : i1hi[1]),
            jlo = has[0] && has[1] ? std::min(i2lo[0], i2lo[1]) : (has[0] ? i2lo[0] : i2lo[1]),
            jhi = has[0] && has[1] ? std::max(i2hi[0], i2hi[1]) : (has[0] ? i2hi[0] : i2hi[1]);

        const len_t offset = ir*N;
        for (len_t j = jlo; j < jhi; j++)
            for (len_t i = ilo; i < ihi; i++) {
                if (!isLocalFace(i, j))
                    continue;

                const len_t idx = offset + j*np1 + i;
                this->_setcoeff(ir, j*np1 + i, (1-w)*y1[idx] + w*y2[idx]);
            }
    }
}

