/**
 * This equation term implements turbulent electron radial diffusion as
 * described by Rechester & Rosenbluth (PRL 40 1978):
 *
 *   Drr = pi*q*R0*(dB/B)^2 * v_||
 *
 * where q = q(r) denotes the tokamak safety factor, R0 is the tokamak major
 * radius, v_|| is the electron parallel speed, and dB is the magnitude by which
 * the magnetic field strength oscillates.
 */


#include <algorithm>
#include "DREAM/Constants.hpp"
#include "DREAM/Equations/Kinetic/RechesterRosenbluthTransport.hpp"
#include "DREAM/Settings/OptionConstants.hpp"
#include "FVM/UnknownQuantityHandler.hpp"


using namespace DREAM;


/**
 * Constructor.
 *
 * grid: Computational grid on which the equation term is defined.
 * dB_B: Object representing the time evolution of the magntiude of
 *       the magnetic field oscillation (dB/B).
 */
RechesterRosenbluthTransport::RechesterRosenbluthTransport(
    FVM::Grid *grid, enum OptionConstants::momentumgrid_type mgtype, FVM::Interpolator1D *dB_B
) : DiffusionTerm(grid), mgtype(mgtype), deltaBOverB(dB_B) {

    SetName("RechesterRosenbluthTransport");
}

/**
 * Destructor.
 */
RechesterRosenbluthTransport::~RechesterRosenbluthTransport() {
    delete this->deltaBOverB;
}


/**
 * Rebuild this equation term.
 */
void RechesterRosenbluthTransport::Rebuild(
    const real_t t, const real_t, FVM::UnknownQuantityHandler*
) {
    const real_t *dB_B = this->deltaBOverB->Eval(t);

    // XXX Here we assume that all momentum grids are the same
    // at all radii...
    const len_t nr = this->grid->GetNr();
    auto mg = this->grid->GetMomentumGrid(0);

    real_t R0  = this->grid->GetRadialGrid()->GetR0();
    const real_t
        *p1 = mg->GetP1(),
        *p2 = mg->GetP2();
    const len_t
        np1 = mg->GetNp1(),
        np2 = mg->GetNp2();

    // Major radius is set to 'inf' in cylindrical geometry.
    // If so, we instead set it to 1 and effectively remove
    // it from the diffusion expression, allowing the user
    // to fully control the diffusion magnitude using the
    // dB/B parameter.
    if (isinf(R0))
        R0 = 1.0;

    for (len_t ir = 0; ir < nr+1; ir++) {
        // Radial face 'ir' is the upper face of cell ir-1 and the lower
        // face of cell ir: set it where this rank owns either cell (once,
        // since the coefficient is accumulated)
        len_t i1lo[2], i1hi[2], i2lo[2], i2hi[2];
        const bool has[2] = {
            ir > 0  && GetLocalRange(ir-1, i1lo[0], i1hi[0], i2lo[0], i2hi[0]),
            ir < nr && GetLocalRange(ir,   i1lo[1], i1hi[1], i2lo[1], i2hi[1])
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

        const real_t q = 1.0;   // TODO (safety factor)
        const real_t *BA_xi = grid->GetBA_xi_fr(ir);
        for (len_t i = ilo; i < ihi; i++) {
            if(np2==1 && (mgtype == OptionConstants::MOMENTUMGRID_TYPE_PXI)){
                if (!isLocalFace(i, 0))
                    continue;
                // ISOTROPIC mode: set pitch-angle averaged diffusion coefficient
                real_t v = Constants::c * p1[i] / sqrt(1+p1[i]*p1[i]);
                real_t xiT = grid->GetRadialGrid()->GetXi0TrappedBoundary_fr(ir);
                real_t geometricFactor = 0.5 * grid->GetRadialGrid()->GetFSA_B_f(ir) * (1 - xiT*xiT);
                real_t D = M_PI * q * R0 * dB_B[ir]*dB_B[ir] * v * geometricFactor;
                Drr(ir,i,0) += D;
            } else {
                for (len_t j = jlo; j < jhi; j++) {
                    if (!isLocalFace(i, j))
                        continue;

                    real_t vpar=1;
                    if (mgtype == OptionConstants::MOMENTUMGRID_TYPE_PXI)
                        vpar = Constants::c * p1[i]*p2[j] / sqrt(1+p1[i]*p1[i]);
                    else if (mgtype == OptionConstants::MOMENTUMGRID_TYPE_PPARPPERP)
                        vpar = Constants::c * p1[i] / sqrt(1 + p1[i]*p1[i] + p2[j]*p2[j]);

                    // Set diffusion coefficient...
                    real_t D = M_PI * q * R0 * dB_B[ir]*dB_B[ir] * fabs(vpar) * BA_xi[j*np1+i];
                    Drr(ir, i, j) += D;
                }
            }
		}
	}
}

