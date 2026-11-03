/**
 * Implementation of the p*nu_s friction term in the kinetic equation.
 */

#include "DREAM/Equations/Kinetic/SlowingDownTerm.hpp"


using namespace DREAM;

/**
 * Constructor.
 */
SlowingDownTerm::SlowingDownTerm(
    FVM::Grid *g, CollisionQuantityHandler *cqh,
    enum OptionConstants::momentumgrid_type mgtype,
    FVM::UnknownQuantityHandler *unknowns,
    bool withKineticIonJacobian
) : FVM::AdvectionTerm(g) {

    SetName("SlowingDownTerm");

    this->gridtype = mgtype;
    this->nuS = cqh->GetNuS();
    AddUnknownForJacobian(unknowns, unknowns->GetUnknownID(OptionConstants::UQTY_N_COLD));
    AddUnknownForJacobian(unknowns, unknowns->GetUnknownID(OptionConstants::UQTY_T_COLD));
    if(withKineticIonJacobian)
        AddUnknownForJacobian(unknowns, unknowns->GetUnknownID(OptionConstants::UQTY_ION_SPECIES));
}

/**
 * Build the coefficients of this advection term.
 */
void SlowingDownTerm::Rebuild(const real_t, const real_t, FVM::UnknownQuantityHandler*){
    real_t *const* nu_s_f1 = nuS->GetValue_f1();
    real_t *const* nu_s_f2 = nuS->GetValue_f2();
  
    bool gridtypePXI        = (gridtype == OptionConstants::MOMENTUMGRID_TYPE_PXI);
    bool gridtypePPARPPERP  = (gridtype == OptionConstants::MOMENTUMGRID_TYPE_PPARPPERP);

    for (len_t ir = 0; ir < nr; ir++) {
        auto *mg = grid->GetMomentumGrid(ir);

        // Cells of this radius owned by this rank
        len_t i1lo, i1hi, i2lo, i2hi;
        if (!GetLocalRange(ir, i1lo, i1hi, i2lo, i2hi))
            continue;
                
        if (gridtypePXI || gridtypePPARPPERP) 
            for (len_t j = i2lo; j < i2hi; j++) 
                for (len_t i = i1lo; i < i1hi+1; i++) 
                    F1(ir, i, j) -= mg->GetP1_f(i) * nu_s_f1[ir][j*(n1[ir]+1)+i];
                
        if (gridtypePPARPPERP) 
            for (len_t j = i2lo; j < i2hi+1; j++) 
                for (len_t i = i1lo; i < i1hi; i++) 
                    F2(ir, i, j) -= mg->GetP2_f(j) * nu_s_f2[ir][j*n1[ir]+i];
                
        if (gridtypePXI){    
            real_t p3nuSAtZero = nuS->GetP3NuSAtZero(ir);
            for (len_t j = i2lo; j < i2hi; j++)
                F1PSqAtZero(ir,j) -= p3nuSAtZero;
        }
    }

}


// Set jacobian of the advection coefficients for this advection term
void SlowingDownTerm::SetPartialAdvectionTerm(len_t derivId, len_t nMultiples){
    ResetDifferentiationCoefficients();

    const len_t nr = grid->GetNr();
 
    // Jacobian of the collision frequency, evaluated on the fly
    real_t *dp3nuSAtZero  = nuS->GetPartialP3NuSAtZero(derivId);
    bool gridtypePXI        = (gridtype == OptionConstants::MOMENTUMGRID_TYPE_PXI);
    bool gridtypePPARPPERP  = (gridtype == OptionConstants::MOMENTUMGRID_TYPE_PPARPPERP);

    // Set partial advection coefficients
    for(len_t n=0; n<nMultiples; n++)
        for (len_t ir = 0; ir < nr; ir++) {
        FVM::MomentumGrid *mg = grid->GetMomentumGrid(ir);
        const len_t np1 = n1[ir];
        const len_t np2 = n2[ir];

        // The collision frequency is only available on the radii of this
        // rank (elsewhere the derivative is left at zero)
        len_t i1lo, i1hi, i2lo, i2hi;
        const bool owned = GetLocalRange(ir, i1lo, i1hi, i2lo, i2hi);

            if (owned && (gridtypePXI || gridtypePPARPPERP))
                for (len_t j = 0; j < np2; j++)
                    for (len_t i = 0; i < np1+1; i++)
                        dF1(ir,i,j,n) = -mg->GetP1_f(i) * nuS->GetUnknownPartialContribution(derivId, FVM::FLUXGRIDTYPE_P1, ir, i, j, n);

            if (owned && gridtypePPARPPERP)
                for (len_t j = 0; j < np2+1; j++)
                    for (len_t i = 0; i < np1; i++)
                        dF2(ir,i,j,n) = -mg->GetP2_f(j) * nuS->GetUnknownPartialContribution(derivId, FVM::FLUXGRIDTYPE_P2, ir, i, j, n);

            if (gridtypePXI == (dp3nuSAtZero != nullptr)) {
                real_t dp3nuS = dp3nuSAtZero[nr*n+ir];
                for (len_t j=0; j<np2; j++)
                    dF1PSqAtZero(ir,j,n) = -dp3nuS;
            }
        }
    delete [] dp3nuSAtZero;
}
