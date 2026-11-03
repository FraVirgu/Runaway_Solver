/**
 * Implementation of the a which handles the construction of
 * collision frequencies as well as their partial derivatives
 * with respect to the unknown quantities.
 * CollisionFrequencies should be rebuilt after their CoulombLogarithms
 * have been rebuilt.
 */

#include "DREAM/Equations/CollisionFrequency.hpp"
#include "DREAM/Constants.hpp"
#include "DREAM/Settings/OptionConstants.hpp"
#include "FVM/UnknownQuantityHandler.hpp"
#include "DREAM/NotImplementedException.hpp"
#include "FVM/FVMException.hpp"
#include <string>
#include "gsl/gsl_sf_bessel.h"

using namespace DREAM;

/**
 * Constructor.
 */
CollisionFrequency::CollisionFrequency(FVM::Grid *g, FVM::UnknownQuantityHandler *u, IonHandler *ih,
                                       CoulombLogarithm *lnLee, CoulombLogarithm *lnLei,
                                       enum OptionConstants::momentumgrid_type mgtype, struct collqty_settings *cqset)
    : CollisionQuantity(g, u, ih, mgtype, cqset)
{
    lnLambdaEE = lnLee;
    lnLambdaEI = lnLei;
}

/**
 * Destructor.
 */
CollisionFrequency::~CollisionFrequency()
{
    DeallocatePartialQuantities();
}

/**
 * Evaluates the collision frequency at radial grid point ir and momentum p,
 * neglecting any contribution from the nonlinear collision operator, using
 * the input collqty_settings object.
 */
real_t CollisionFrequency::evaluateAtP(len_t ir, real_t p, collqty_settings *inSettings)
{
    bool isPartiallyScreened = (inSettings->collfreq_type == OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_PARTIALLY_SCREENED || inSettings->collfreq_type == OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_PARTIALLY_SCREENED_WALKOWIAK);
    bool isNonScreened = (inSettings->collfreq_type == OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_NON_SCREENED);
    bool isBrems = (inSettings->bremsstrahlung_mode != OptionConstants::EQTERM_BREMSSTRAHLUNG_MODE_NEGLECT);
    real_t ntarget = GetNTarget(ir, isNonScreened);
    real_t preFact = evaluatePreFactorAtP(p, inSettings->collfreq_mode);
    real_t lnLee = lnLambdaEE->evaluateAtP(ir, p, inSettings);
    real_t lnLei = lnLambdaEI->evaluateAtP(ir, p, inSettings);

    // Add electron contribution to collision frequency
    real_t collFreq = lnLee * evaluateElectronTermAtP(ir, p, inSettings->collfreq_mode) * ntarget;

    len_t ind;
    // Add ion contribution; SlowingDownFrequency doesn't have one and will skip this step
    if (hasIonTerm)
    {
        for (len_t iz = 0; iz < nZ; iz++)
            for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
            {
                ind = ionIndex[iz][Z0];
                len_t Zfact = Z0 * Z0;
                if (isNonScreened)
                    Zfact = Zs[iz] * Zs[iz];
                collFreq += lnLei * Zfact * evaluateIonTermAtP(iz, Z0, p) * ionDensities[ir][ind];
            }
    }
    // Add screening contribution
    if (isPartiallyScreened)
        for (len_t iz = 0; iz < nZ; iz++)
            for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
            {
                ind = ionIndex[iz][Z0];
                collFreq += evaluateScreenedTermAtP(iz, Z0, p, inSettings->collfreq_type) * ionDensities[ir][ind];
            }
    // Add stopping contribution
    if (isPartiallyScreened)
        for (len_t iz = 0; iz < nZ; iz++)
            for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
            {
                ind = ionIndex[iz][Z0];
                collFreq += evaluateStoppingTermAtP(iz, Z0, p, inSettings->collfreq_mode) * ionDensities[ir][ind];
            }
    collFreq *= preFact;

    // Add Bremsstrahlung contribution
    if (isBrems)
        for (len_t iz = 0; iz < nZ; iz++)
            for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
            {
                ind = ionIndex[iz][Z0];
                collFreq += evaluateBremsstrahlungTermAtP(iz, Z0, p, inSettings->bremsstrahlung_mode) * ionDensities[ir][ind];
            }

    return collFreq;
}

/**
 *  Calculates and stores partial terms which depend on unknown quantities (density and temperature).
 */
void CollisionFrequency::RebuildPlasmaDependentTerms()
{
    len_t indZ;
    for (len_t iz = 0; iz < nZ; iz++)
        for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
            for (len_t ir = 0; ir < nr; ir++)
            {
                indZ = ionIndex[iz][Z0];
                ionDensities[ir][indZ] = ionHandler->GetIonDensity(ir, iz, Z0);
            }

    for (len_t ir = 0; ir < nr; ir++)
    {
        const real_t Theta = unknowns->GetUnknownData(id_Tcold)[ir] / Constants::mc2inEV;
        K0Scaled[ir] = evaluateExp1OverThetaK(Theta, 0.0);
        K1Scaled[ir] = evaluateExp1OverThetaK(Theta, 1.0);
        K2Scaled[ir] = evaluateExp1OverThetaK(Theta, 2.0);
    }

    if (collQtySettings->collfreq_mode == OptionConstants::COLLQTY_COLLISION_FREQUENCY_MODE_FULL)
        InitializeGSLWorkspace();
    if (!buildOnlyF1F2)
    {
        setElectronTerm(nColdTerm, mg->GetP(), nr, np1, np2_store);
        setElectronTerm(nColdTerm_fr, mg->GetP(), nr /*+1*/, np1, np2_store);
    }
    setElectronTerm(nColdTerm_f1, mg->GetP_f1(), nr, np1 + 1, np2_store);
    setElectronTerm(nColdTerm_f2, mg->GetP_f2(), nr, np1, np2_store + 1);
    SetLnLambdaPartialNi();
}

/**
 * Calculates and stores the derivatives of the Coulomb logarithms with
 * respect to the ion densities (at p=0), on the radii owned by this rank.
 */
void CollisionFrequency::SetLnLambdaPartialNi()
{
    for (len_t ir = irlo; ir < irhi; ir++)
        for (len_t iz = 0; iz < nZ; iz++)
            for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
            {
                len_t indZ = ionIndex[iz][Z0];
                lnLEE_partialNi[ir][indZ] = lnLambdaEE->evaluatePartialAtP(ir, 0, id_ni, indZ);
                lnLEI_partialNi[ir][indZ] = lnLambdaEI->evaluatePartialAtP(ir, 0, id_ni, indZ);
            }
}

/**
 * Calculates and stores all quantities that do not depend on momentum. Is to be used instead of Rebuild when
 * you only want to use the CollisionFrequency::evaluateAtP function (such as in RunawayFluid).
 */
void CollisionFrequency::RebuildRadialTerms()
{
    if (gridRebuilt)
    {
        DeallocateRadialQuantities();
        nr = rGrid->GetNr();
        nZ = ionHandler->GetNZ();
        nzs = ionHandler->GetNzs();
        AllocateRadialQuantities();
    }
    len_t indZ;
    const len_t *ZAtomicCharge = ionHandler->GetZs();
    for (len_t iz = 0; iz < nZ; iz++)
    {
        Zs[iz] = ZAtomicCharge[iz];
        for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
        {
            indZ = ionHandler->GetIndex(iz, Z0);
            ionIndex[iz][Z0] = indZ;
            atomicParameter[indZ] = GetAtomicParameter(iz, Z0);
            for (len_t ir = 0; ir < nr; ir++)
                ionDensities[ir][indZ] = ionHandler->GetIonDensity(ir, iz, Z0);
        }
    }
    if (collQtySettings->collfreq_mode == OptionConstants::COLLQTY_COLLISION_FREQUENCY_MODE_FULL)
        InitializeGSLWorkspace();
}

/**
 * Allocates quantities involved in RebuildRadialTerms()
 */
void CollisionFrequency::AllocateRadialQuantities()
{
    //    DeallocateRadialQuantities();
    InitializeGSLWorkspace();
    Zs = new real_t[nZ];
    ionIndex = new real_t *[nZ];
    ionDensities = new real_t *[nr];
    atomicParameter = new real_t[nzs];

    for (len_t iz = 0; iz < nZ; iz++)
        ionIndex[iz] = new real_t[ionHandler->GetZ(iz) + 1];
    for (len_t ir = 0; ir < nr; ir++)
        ionDensities[ir] = new real_t[nzs];
}

/**
 * Deallocates quantities involved in RebuildRadialTerms()
 */
void CollisionFrequency::DeallocateRadialQuantities()
{
    if (Zs != nullptr)
    {
        delete[] Zs;
        delete[] atomicParameter;

        for (len_t iz = 0; iz < nZ; iz++)
            delete[] ionIndex[iz];
        for (len_t ir = 0; ir < nr; ir++)
            delete[] ionDensities[ir];
        delete[] ionIndex;
        delete[] ionDensities;
    }
}

/**
 * Calculates and stores partial contributions that only depend on the grid. If using P-Xi grid,
 * only store momentum dependent quantities on a size np1 array.
 */
void CollisionFrequency::RebuildConstantTerms()
{
    const len_t *ZAtomicCharge = ionHandler->GetZs();
    len_t indZ;
    for (len_t iz = 0; iz < nZ; iz++)
    {
        Zs[iz] = ZAtomicCharge[iz];
        for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
        {
            indZ = ionHandler->GetIndex(iz, Z0);
            ionIndex[iz][Z0] = indZ;
            ionIZ[indZ] = iz;
            ionZ0[indZ] = Z0;
            atomicParameter[indZ] = GetAtomicParameter(iz, Z0);
        }
    }

    if (!buildOnlyF1F2)
    {
        setPreFactor(preFactor, mg->GetP(), np1, np2_store);
        setPreFactor(preFactor_fr, mg->GetP(), np1, np2_store);
        setIonTerm(ionTerm, mg->GetP(), np1, np2_store);
        setIonTerm(ionTerm_fr, mg->GetP(), np1, np2_store);
        if (isBrems)
        {
            setBremsTerm(bremsTerm, mg->GetP(), np1, np2_store);
            setBremsTerm(bremsTerm_fr, mg->GetP(), np1, np2_store);
        }
        if (isPartiallyScreened)
        {
            setScreenedTerm(screenedTerm, mg->GetP(), np1, np2_store);
            setScreenedTerm(screenedTerm_fr, mg->GetP(), np1, np2_store);
            setStoppingTerm(stoppingTerm, mg->GetP(), np1, np2_store);
            setStoppingTerm(stoppingTerm_fr, mg->GetP(), np1, np2_store);
        }
    }
    setPreFactor(preFactor_f1, mg->GetP_f1(), np1 + 1, np2_store);
    setPreFactor(preFactor_f2, mg->GetP_f2(), np1, np2_store + 1);
    setIonTerm(ionTerm_f1, mg->GetP_f1(), np1 + 1, np2_store);
    setIonTerm(ionTerm_f2, mg->GetP_f2(), np1, np2_store + 1);
    if (isBrems)
    {
        setBremsTerm(bremsTerm_f1, mg->GetP_f1(), np1 + 1, np2_store);
        setBremsTerm(bremsTerm_f2, mg->GetP_f2(), np1, np2_store + 1);
    }
    if (isPartiallyScreened)
    {
        setScreenedTerm(screenedTerm_f1, mg->GetP_f1(), np1 + 1, np2_store);
        setScreenedTerm(screenedTerm_f2, mg->GetP_f2(), np1, np2_store + 1);
        setStoppingTerm(stoppingTerm_f1, mg->GetP_f1(), np1 + 1, np2_store);
        setStoppingTerm(stoppingTerm_f2, mg->GetP_f2(), np1, np2_store + 1);
    }
    if (isNonlinear)
        calculateIsotropicNonlinearOperatorMatrix();
}

/**
 * Collects the stored factors of the collision frequency on the given
 * flux grid, from which the partial contributions are evaluated.
 */
CollisionFrequency::gridTerms CollisionFrequency::GetGridTerms(FVM::fluxGridType fluxGridType) const
{
    gridTerms t;
    t.lnLee = lnLambdaEE->GetValue(fluxGridType);
    t.lnLei = lnLambdaEI->GetValue(fluxGridType);
    t.n1 = np1 + (fluxGridType == FVM::FLUXGRIDTYPE_P1);
    t.n2 = np2 + (fluxGridType == FVM::FLUXGRIDTYPE_P2);
    // On a p-xi grid the momentum-dependent factors are stored for a single
    // xi (two on the p2 flux grid)
    const len_t n2Store = isPXiGrid ? 1 + t.n2 - np2 : t.n2;
    t.zStride = t.n1 * n2Store;

    switch (fluxGridType)
    {
    case FVM::FLUXGRIDTYPE_DISTRIBUTION:
        t.nColdTerm = nColdTerm;
        t.preFactor = preFactor;
        t.ionTerm = ionTerm;
        t.screenedTerm = screenedTerm;
        t.stoppingTerm = stoppingTerm;
        t.bremsTerm = bremsTerm;
        break;
    case FVM::FLUXGRIDTYPE_RADIAL:
        t.nColdTerm = nColdTerm_fr;
        t.preFactor = preFactor_fr;
        t.ionTerm = ionTerm_fr;
        t.screenedTerm = screenedTerm_fr;
        t.stoppingTerm = stoppingTerm_fr;
        t.bremsTerm = bremsTerm_fr;
        break;
    case FVM::FLUXGRIDTYPE_P1:
        t.nColdTerm = nColdTerm_f1;
        t.preFactor = preFactor_f1;
        t.ionTerm = ionTerm_f1;
        t.screenedTerm = screenedTerm_f1;
        t.stoppingTerm = stoppingTerm_f1;
        t.bremsTerm = bremsTerm_f1;
        break;
    case FVM::FLUXGRIDTYPE_P2:
        t.nColdTerm = nColdTerm_f2;
        t.preFactor = preFactor_f2;
        t.ionTerm = ionTerm_f2;
        t.screenedTerm = screenedTerm_f2;
        t.stoppingTerm = stoppingTerm_f2;
        t.bremsTerm = bremsTerm_f2;
        break;
    default:
        throw FVM::FVMException("Invalid fluxGridType");
    }
    return t;
}

/**
 * Partial derivative of the collision frequency with respect to n_cold
 * (ignoring variations with lnLambda) in cell 'pind' of radius 'ir'.
 * 'pindStore' is the index of the cell in the momentum-dependent factors
 * (i on a p-xi grid, pind otherwise).
 */
real_t CollisionFrequency::evaluateNColdPartial(const gridTerms &t, len_t ir, len_t pind, len_t pindStore) const
{
    return t.nColdTerm[ir][pindStore] * t.preFactor[pindStore] * t.lnLee[ir][pind];
}

/**
 * Partial derivative of the collision frequency with respect to the ion
 * density of charge state 'indZ' in cell 'pind' of radius 'ir', including
 * the variation of lnLambda with the ion density. That variation is also
 * returned alone in 'lnLContrib', so that 'partial - lnLContrib' is the
 * coefficient of the (linear) ion contribution to the frequency.
 *
 * The terms are added in the same order as when they were stored, so that
 * the result is identical.
 */
real_t CollisionFrequency::evaluateNiPartial(
    const gridTerms &t, len_t ir, len_t pind, len_t pindStore, len_t indZ,
    real_t ntarget, real_t &lnLContrib) const
{
    const len_t iz = ionIZ[indZ];
    const len_t Z0 = ionZ0[indZ];
    const len_t zind = indZ * t.zStride + pindStore;

    // variation of lnLambda_ee with the ion density
    real_t electronTerm = ntarget * t.nColdTerm[ir][pindStore] * t.preFactor[pindStore];
    lnLContrib = electronTerm * lnLEE_partialNi[ir][indZ];
    real_t partQty = lnLContrib;

    if (hasIonTerm)
    {
        real_t partContrib = t.preFactor[pindStore] * t.lnLei[ir][pindStore];
        real_t DpartContrib = ionDensities[ir][indZ] * t.preFactor[pindStore] * lnLEI_partialNi[ir][indZ];
        len_t Zfact;
        if (isNonScreened)
            Zfact = Zs[iz] * Zs[iz] * t.ionTerm[zind];
        else
            Zfact = Z0 * Z0 * t.ionTerm[zind];
        real_t lnLContribIon = Zfact * DpartContrib;
        lnLContrib += lnLContribIon;
        partQty += Zfact * partContrib + lnLContribIon;
    }
    if (isBrems)
        partQty += t.bremsTerm[zind];
    if (isNonScreened)
        partQty += (Zs[iz] - Z0) * evaluateNColdPartial(t, ir, pind, pindStore);
    else if (isPartiallyScreened)
        partQty += t.preFactor[pindStore] * (t.screenedTerm[zind] + t.stoppingTerm[zind]);

    return partQty;
}

/**
 * Assembles the collision frequency on the radii owned by this rank from
 * its partial contributions, which are evaluated on the fly.
 */
void CollisionFrequency::AssembleQuantity(real_t **&collisionQuantity, len_t /*nr*/, len_t np1, len_t np2, enum FVM::fluxGridType fluxGridType)
{
    real_t collQty;
    real_t *ncold = unknowns->GetUnknownData(id_ncold);
    const gridTerms t = GetGridTerms(fluxGridType);

    // only the radii owned by this rank are allocated
    for (len_t ir = irlo; ir < irhi; ir++)
    {
        real_t ntarget = GetNTarget(ir, isNonScreened);
        for (len_t j = 0; j < np2; j++)
            for (len_t i = 0; i < np1; i++)
            {
                len_t pind = np1 * j + i;
                len_t pindStore = isPXiGrid ? i : pind;
                // the collision frequencies are linear in ncold
                collQty = ncold[ir] * evaluateNColdPartial(t, ir, pind, pindStore);
                for (len_t indZ = 0; indZ < nzs; indZ++)
                {
                    real_t lnLContrib;
                    real_t ionContrib = evaluateNiPartial(t, ir, pind, pindStore, indZ, ntarget, lnLContrib);
                    // when subtracting the lnLambda terms, the collision frequencies are linear in ion densities
                    collQty += ionDensities[ir][indZ] * (ionContrib - lnLContrib);
                }
                collisionQuantity[ir][pind] = collQty;
            }
    }
}

/**
 * Partial derivative of the collision frequency with respect to the unknown
 * quantity with ID id_unknown (ignoring variations with lnLambda) in cell
 * (i,j) of radius 'ir' on the given flux grid; 'n' is the multiple (the
 * charge state for the ion densities). It is evaluated on the fly from the
 * stored factors, so 'ir' must be a radius owned by this rank.
 *
 * The derivatives with respect to T_cold and f_hot (nonlinear operator) are
 * not available: they are only needed by the Jacobian of SolverNonLinear.
 */
real_t CollisionFrequency::GetUnknownPartialContribution(
    len_t id_unknown, FVM::fluxGridType fluxGridType, len_t ir, len_t i, len_t j, len_t n)
{
    if (id_unknown == id_Tcold)
        throw FVM::FVMException("CollisionFrequency: the partial derivative with respect to T_cold is not available.");
    else if (id_unknown != id_ncold && id_unknown != id_ni)
    {
        if (id_unknown == unknowns->GetUnknownID(OptionConstants::UQTY_F_HOT))
            throw FVM::FVMException("CollisionFrequency: the nonlinear partial derivative with respect to f_hot is not available.");
        return 0;
    }

    const gridTerms t = GetGridTerms(fluxGridType);
    const len_t pind = t.n1 * j + i;
    const len_t pindStore = isPXiGrid ? i : pind;
    if (id_unknown == id_ncold)
        return evaluateNColdPartial(t, ir, pind, pindStore);

    real_t lnLContrib;
    return evaluateNiPartial(t, ir, pind, pindStore, n, GetNTarget(ir, isNonScreened), lnLContrib);
}

/**
 * Calculates and stores the momentum-dependent prefactor to the collision frequencies.
 */
void CollisionFrequency::setPreFactor(real_t *&preFactor, const real_t *pIn, len_t np1, len_t np2)
{
    len_t N = np1 * np2;
    for (len_t pind = 0; pind < N; pind++)
        preFactor[pind] = evaluatePreFactorAtP(pIn[pind], collQtySettings->collfreq_mode);
}

/**
 * Calculates and stores the ion contribution to the collision frequency.
 */
void CollisionFrequency::setIonTerm(real_t *&ionTerm, const real_t *pIn, len_t np1, len_t np2)
{
    if (!hasIonTerm)
        return;
    real_t p;
    len_t ind, pind;
    len_t N = np1 * np2;
    for (len_t i = 0; i < np1; i++)
        for (len_t j = 0; j < np2; j++)
        {
            pind = np1 * j + i;
            p = pIn[pind];
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    ionTerm[ind * N + pind] = evaluateIonTermAtP(iz, Z0, p);
                }
        }
}

/**
 * Calculates and stores the partially-screened contribution to the collision frequency.
 */
void CollisionFrequency::setScreenedTerm(real_t *&screenedTerm, const real_t *pIn, len_t np1, len_t np2)
{
    len_t ind;
    len_t N = np1 * np2;
    if (isPXiGrid)
        for (len_t i = 0; i < np1; i++)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    real_t screenedAtP = evaluateScreenedTermAtP(iz, Z0, pIn[i], collQtySettings->collfreq_type);
                    for (len_t j = 0; j < np2; j++)
                        screenedTerm[ind * N + np1 * j + i] = screenedAtP;
                }
    else
        for (len_t pind = 0; pind < N; pind++)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    screenedTerm[ind * N + pind] = evaluateScreenedTermAtP(iz, Z0, pIn[pind], collQtySettings->collfreq_type);
                }
}

/**
 * Calculates and stores the stopping contribution to the collision frequency.
 */
void CollisionFrequency::setStoppingTerm(real_t *&stoppingTerm, const real_t *pIn, len_t np1, len_t np2)
{
    len_t ind;
    len_t N = np1 * np2;
    if (isPXiGrid)
        for (len_t i = 0; i < np1; i++)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    real_t stoppingAtP = evaluateStoppingTermAtP(iz, Z0, pIn[i], collQtySettings->collfreq_mode);
                    for (len_t j = 0; j < np2; j++)
                        stoppingTerm[ind * N + np1 * j + i] = stoppingAtP;
                }
    else
        for (len_t pind = 0; pind < N; pind++)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    stoppingTerm[ind * N + pind] = evaluateStoppingTermAtP(iz, Z0, pIn[pind], collQtySettings->collfreq_mode);
                }
}

/**
 * Calculates and stores the bremsstrahlung contribution to the collision frequency.
 */
void CollisionFrequency::setBremsTerm(real_t *&bremsTerm, const real_t *pIn, len_t np1, len_t np2)
{
    len_t ind;
    len_t N = np1 * np2;

    if (isPXiGrid)
    {
        for (len_t i = 0; i < np1; i++)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    real_t bremsAtP = evaluateBremsstrahlungTermAtP(iz, Z0, pIn[i], collQtySettings->bremsstrahlung_mode);
                    for (len_t j = 0; j < np2; j++)
                        bremsTerm[ind * N + np1 * j + i] = bremsAtP;
                }
    }
    else
    {
        for (len_t pind = 0; pind < N; pind++)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    ind = ionIndex[iz][Z0];
                    bremsTerm[ind * N + pind] = evaluateBremsstrahlungTermAtP(iz, Z0, pIn[pind], collQtySettings->bremsstrahlung_mode);
                }
    }
}

/**
 * Calculates and stores the free-electron contribution to the collision frequency.
 */
void CollisionFrequency::setElectronTerm(real_t **&nColdTerm, const real_t *pIn, len_t nr, len_t np1, len_t np2)
{
    // only the radii owned by this rank are used
    if (isPXiGrid)
        for (len_t i = 0; i < np1; i++)
            for (len_t ir = irlo; ir < irhi; ir++)
            {
                real_t electronTerm = evaluateElectronTermAtP(ir, pIn[i], collQtySettings->collfreq_mode);
                for (len_t j = 0; j < np2; j++)
                    nColdTerm[ir][np1 * j + i] = electronTerm;
            }
    else
        for (len_t pind = 0; pind < np1 * np2; pind++)
            for (len_t ir = irlo; ir < irhi; ir++)
                nColdTerm[ir][pind] = evaluateElectronTermAtP(ir, pIn[pind], collQtySettings->collfreq_mode);
}

// PSI FUNCTIONS FOR EVALUATION OF "FULL" COLLFREQ_MODE
// Is the relativistic generalisation of the Chandrasekhar functions,
// appearing in all collision frequencies.
real_t CollisionFrequency::psi0Integrand(real_t s, void *params)
{
    real_t Theta = *(real_t *)params;
    real_t gs = sqrt(1 + s * s);
    real_t gsMinusOne = s * s / (1 + gs); // = gs - 1
    return exp(-gsMinusOne / Theta) / gs;
}

real_t CollisionFrequency::psi1Integrand(real_t s, void *params)
{
    real_t Theta = *(real_t *)params;
    real_t gs = sqrt(1 + s * s);
    real_t gsMinusOne = s * s / (1 + gs); // = gs - 1
    return exp(-gsMinusOne / Theta);
}
real_t CollisionFrequency::psi2Integrand(real_t s, void *params)
{
    real_t Theta = *(real_t *)params;
    real_t gs = sqrt(1 + s * s);
    real_t gsMinusOne = s * s / (1 + gs); // = gs - 1
    return exp(-gsMinusOne / Theta) * gs;
}

/**
 * Evaluates the second-order asymptotic expansion of the Psi_n functions in
 * the low-energy or low-temperature limit p<<1 or Theta<<1.
 * Documented in doc/notes/psi0psi1evaluation
 */
real_t CollisionFrequency::evaluatePsiLowenergyLimit(len_t n, real_t p, real_t Theta)
{
    real_t gamma = sqrt(1 + p * p);
    real_t x = sqrt((gamma - 1) / Theta);
    real_t x2 = x * x;
    real_t erfAtX = erf(x);
    real_t Theta2 = Theta * Theta;
    real_t O1 = (4.0 * n - 1.0);               // (4n-1)
    real_t O2 = 2.0 * n * n - 3.0 * n + 0.375; // 2n^2 - 3n + 3/8
    real_t psi_n = (1 + Theta * 0.125 * O1 + Theta2 * 0.1875 * O2) * 0.5 * M_SQRTPI * erfAtX;
    psi_n -= 0.125 * Theta * x * exp(-x2) * (O1 + Theta * O2 * 0.5 * (3 + 2 * x2));
    psi_n *= sqrt(2 * Theta);

    return psi_n;
}

/**
 * Returns true if we are in the superthermal limit
 * where the appropriate asymptotic expansion for the
 * Psi functions is applicable.
 * Defined as E = sqrt(1+p^2)-1 > energyThreshold.
 * At 10 thermal energies, the relative errors in the
 * asymptotic expansion are less than 1e-8
 */
bool isSuperthermalLimit(real_t p, real_t Theta)
{
    real_t superthermalEnergyThreshold = 10.0 * Theta;
    return (p * p > superthermalEnergyThreshold * (superthermalEnergyThreshold + 2));
}

/**
 * Returns true if we are in the low-energy limit
 * where the appropriate asymptotic expansion for the
 * Psi functions is applicable.
 * Satisfied either if temperature is sufficiently low
 * (in which case it is applicable for all energies)
 * or if the momentum is non-relativistic (then valid
 * for all temperatures). The conditions (p<0.15 and Theta<0.005)
 * ensure relative errors <1e-8 when using the low-energy
 * asymptotic expansion for Psi.
 */
bool isLowEnergyLimit(real_t p, real_t Theta)
{
    return (p < 0.15) || (Theta < 0.005);
}

/**
 * Evaluates the Psi0 thermal collision frequency function.
 */
real_t CollisionFrequency::evaluatePsi0(len_t ir, real_t p)
{
    real_t *T_cold = unknowns->GetUnknownData(id_Tcold);
    real_t Theta = T_cold[ir] / Constants::mc2inEV;

    bool superthermalLimit = isSuperthermalLimit(p, Theta);
    bool lowenergyLimit = isLowEnergyLimit(p, Theta);
    if (superthermalLimit)
    {
        // asymptotic expansion described in doc/notes/psi0psi1evaluation
        real_t gamma = sqrt(1 + p * p);
        real_t gammaMinusOne = p * p / (gamma + 1); // = gamma-1
        real_t arg = gammaMinusOne / Theta;
        real_t Term0 = K0Scaled[ir];
        if (arg > 700.0) // exp(-arg) underflows to < 1e-304, negligible vs Term0
            return Term0;
        real_t expTerm = exp(-arg);
        real_t Term1 = -1.0 / p * expTerm;
        real_t Term2 = gamma / (p * p * p) * expTerm;
        return Term0 + Theta * Term1 + Theta * Theta * Term2;
    }
    else if (lowenergyLimit)
    {
        return evaluatePsiLowenergyLimit(0, p, Theta);
    }
    else
    {
        gsl_function F;
        F.function = &(CollisionFrequency::psi0Integrand);
        F.params = &Theta;
        real_t psi0int, error;

        real_t epsabs = 0, epsrel = 1e-8, lim = gsl_ad_w->limit;
        gsl_integration_qag(&F, 0, p, epsabs, epsrel, lim, QAG_KEY, gsl_ad_w, &psi0int, &error);
        return psi0int;
    }
}

/**
 * Evaluates the Psi1 thermal collision frequency function.
 */
real_t CollisionFrequency::evaluatePsi1(len_t ir, real_t p)
{
    real_t *T_cold = unknowns->GetUnknownData(id_Tcold);
    real_t Theta = T_cold[ir] / Constants::mc2inEV;

    bool superthermalLimit = isSuperthermalLimit(p, Theta);
    bool lowenergyLimit = isLowEnergyLimit(p, Theta);
    if (superthermalLimit)
    {
        // asymptotic expansion described in doc/notes/psi0psi1evaluation
        real_t gamma = sqrt(1 + p * p);
        real_t gammaMinusOne = p * p / (gamma + 1); // = gamma-1
        real_t arg = gammaMinusOne / Theta;
        real_t Term0 = K1Scaled[ir];
        if (arg > 700.0) // exp(-arg) underflows to < 1e-304, negligible vs Term0
            return Term0;
        real_t expTerm = exp(-arg);
        real_t Term1 = -gamma / p * expTerm;
        real_t Term2 = 1.0 / (p * p * p) * expTerm;
        return Term0 + Theta * Term1 + Theta * Theta * Term2;
    }
    else if (lowenergyLimit)
    {
        return evaluatePsiLowenergyLimit(1, p, Theta);
    }
    else
    {
        gsl_function F;
        F.function = &(CollisionFrequency::psi1Integrand);
        F.params = &Theta;
        real_t psi1int, error;

        real_t epsabs = 0, epsrel = 1e-8, lim = gsl_ad_w->limit;
        gsl_integration_qag(&F, 0, p, epsabs, epsrel, lim, QAG_KEY, gsl_ad_w, &psi1int, &error);
        return psi1int;
    }
}

real_t CollisionFrequency::evaluatePsi2(len_t ir, real_t p)
{
    real_t *T_cold = unknowns->GetUnknownData(id_Tcold);
    real_t Theta = T_cold[ir] / Constants::mc2inEV;

    bool superthermalLimit = isSuperthermalLimit(p, Theta);
    bool lowenergyLimit = isLowEnergyLimit(p, Theta);
    if (superthermalLimit)
    {
        // asymptotic expansion described in doc/notes/psi0psi1evaluation
        real_t gamma = sqrt(1 + p * p);
        real_t gammaMinusOne = p * p / (gamma + 1); // = gamma-1
        real_t arg = gammaMinusOne / Theta;
        real_t Term0 = K0Scaled[ir];
        if (arg > 700.0) // exp(-arg) underflows to < 1e-304, negligible vs the K1Scaled term
            return Term0 + Theta * K1Scaled[ir];
        real_t expTerm = exp(-arg);
        real_t Term1 = K1Scaled[ir] - gamma * gamma / p * expTerm;
        real_t Term2 = -gamma * (gamma * gamma - 2) / (p * p * p) * expTerm;
        return Term0 + Theta * Term1 + Theta * Theta * Term2;
    }
    else if (lowenergyLimit)
    {
        return evaluatePsiLowenergyLimit(2, p, Theta);
    }
    else
    {
        gsl_function F;
        F.function = &(CollisionFrequency::psi2Integrand);
        F.params = &Theta;
        real_t psi2int, error;

        real_t epsabs = 0, epsrel = 1e-8, lim = gsl_ad_w->limit;
        gsl_integration_qags(&F, 0, p, epsabs, epsrel, lim, gsl_ad_w, &psi2int, &error);
        return psi2int;
    }
}

/**
 * Evaluates e^x K_n(x), with K_n the exponentially decreasing modified bessel function.
 */
real_t CollisionFrequency::evaluateExp1OverThetaK(real_t Theta, real_t n)
{
    return gsl_sf_bessel_Kn_scaled(n, 1.0 / Theta);
}

/**
 * Evaluate the number density of target electrons, corresponding
 * to the number of cold electrons unless using 'NONSCREENED', in which
 * case we add the bound electrons.
 */
real_t CollisionFrequency::GetNTarget(len_t ir, bool isNonScreened)
{
    real_t *ncold = unknowns->GetUnknownData(id_ncold);
    real_t ntarget = ncold[ir];
    if (isNonScreened)
        ntarget += ionHandler->GetBoundElectronDensity(ir);
    if (ntarget < 0) // resolve roundoff error
        ntarget = 0;
    return ntarget;
}

/**
 * Allocates quantities which will be used in the calculation of the collision frequencies.
 */
void CollisionFrequency::AllocatePartialQuantities()
{
    DeallocatePartialQuantities();
    InitializeGSLWorkspace();
    Zs = new real_t[nZ];
    ionIndex = new real_t *[nZ];
    ionDensities = new real_t *[nr];
    atomicParameter = new real_t[nzs];

    K0Scaled = new real_t[nr];
    K1Scaled = new real_t[nr];
    K2Scaled = new real_t[nr];

    for (len_t iz = 0; iz < nZ; iz++)
        ionIndex[iz] = new real_t[ionHandler->GetZ(iz) + 1];
    for (len_t ir = 0; ir < nr; ir++)
        ionDensities[ir] = new real_t[nzs];

    ionIZ = new len_t[nzs];
    ionZ0 = new len_t[nzs];
    // only the radii owned by this rank are allocated
    lnLEE_partialNi = new real_t *[nr];
    lnLEI_partialNi = new real_t *[nr];
    for (len_t ir = 0; ir < nr; ir++)
    {
        const bool owned = (ir >= irlo && ir < irhi);
        lnLEE_partialNi[ir] = owned ? new real_t[nzs] : nullptr;
        lnLEI_partialNi[ir] = owned ? new real_t[nzs] : nullptr;
    }

    if (!buildOnlyF1F2)
    {
        preFactor = new real_t[np1 * np2_store];
        preFactor_fr = new real_t[np1 * np2_store];
        if (hasIonTerm)
        {
            ionTerm = new real_t[nzs * np1 * np2_store];
            ionTerm_fr = new real_t[nzs * np1 * np2_store];
        }

        if (isPartiallyScreened)
        {
            screenedTerm = new real_t[nzs * np1 * np2_store];
            screenedTerm_fr = new real_t[nzs * np1 * np2_store];
            stoppingTerm = new real_t[nzs * np1 * np2_store];
            stoppingTerm_fr = new real_t[nzs * np1 * np2_store];
        }
        if (isBrems)
        {
            bremsTerm = new real_t[nzs * np1 * np2_store];
            bremsTerm_fr = new real_t[nzs * np1 * np2_store];
        }
        // only the radii owned by this rank are allocated
        nColdTerm = new real_t *[nr];
        nColdTerm_fr = new real_t *[nr + 1];
        for (len_t ir = 0; ir < nr; ir++)
            nColdTerm[ir] = (ir >= irlo && ir < irhi) ? new real_t[np1 * np2_store] : nullptr;
        for (len_t ir = 0; ir < nr + 1; ir++)
            nColdTerm_fr[ir] = (ir >= irlo && ir < irhi) ? new real_t[np1 * np2_store] : nullptr;
    }
    preFactor_f1 = new real_t[(np1 + 1) * np2_store];
    preFactor_f2 = new real_t[np1 * (np2_store + 1)];
    if (hasIonTerm)
    {
        ionTerm_f1 = new real_t[nzs * (np1 + 1) * np2_store];
        ionTerm_f2 = new real_t[nzs * np1 * (np2_store + 1)];
    }
    if (isPartiallyScreened)
    {
        screenedTerm_f1 = new real_t[nzs * (np1 + 1) * np2_store];
        screenedTerm_f2 = new real_t[nzs * np1 * (np2_store + 1)];
        stoppingTerm_f1 = new real_t[nzs * (np1 + 1) * np2_store];
        stoppingTerm_f2 = new real_t[nzs * np1 * (np2_store + 1)];
    }
    if (isBrems)
    {
        bremsTerm_f1 = new real_t[nzs * (np1 + 1) * np2_store];
        bremsTerm_f2 = new real_t[nzs * np1 * (np2_store + 1)];
    }
    // only the radii owned by this rank are allocated
    nColdTerm_f1 = new real_t *[nr];
    nColdTerm_f2 = new real_t *[nr];
    for (len_t ir = 0; ir < nr; ir++)
    {
        const bool owned = (ir >= irlo && ir < irhi);
        nColdTerm_f1[ir] = owned ? new real_t[(np1 + 1) * np2_store] : nullptr;
        nColdTerm_f2[ir] = owned ? new real_t[np1 * (np2_store + 1)] : nullptr;
    }

    if (isNonlinear)
    {
        nonlinearMat = new real_t *[np1 + 1]; // multiply matrix by f lnLc to get p*nu_s on p flux grid
        for (len_t i = 0; i < np1 + 1; i++)
            nonlinearMat[i] = new real_t[np1];

        const real_t *p = mg->GetP1();
        trapzWeights = new real_t[np1];
        for (len_t i = 1; i < np1 - 1; i++)
            trapzWeights[i] = (p[i + 1] - p[i - 1]) / 2;
    }
}

/**
 * Deallocator
 */
void CollisionFrequency::DeallocatePartialQuantities()
{
    DeallocateGSL();
    if (Zs != nullptr)
    {
        delete[] Zs;
        for (len_t iz = 0; iz < nZ; iz++)
            delete[] ionIndex[iz];
        for (len_t ir = 0; ir < nr; ir++)
            delete[] ionDensities[ir];

        delete[] ionIndex;
        delete[] ionDensities;

        this->Zs = nullptr;
    }
    if (K0Scaled != nullptr)
    {
        delete[] K0Scaled;
        delete[] K1Scaled;
        delete[] K2Scaled;

        this->K0Scaled = nullptr;
    }
    if (preFactor != nullptr)
    {
        delete[] preFactor;
        delete[] preFactor_fr;

        this->preFactor = nullptr;
    }
    if (ionTerm != nullptr)
    {
        delete[] ionTerm;
        delete[] ionTerm_fr;

        this->ionTerm = nullptr;
    }
    if (preFactor_f1 != nullptr)
    {
        delete[] preFactor_f1;
        delete[] preFactor_f2;

        this->preFactor_f1 = nullptr;
    }
    if (ionTerm_f1 != nullptr)
    {
        delete[] ionTerm_f1;
        delete[] ionTerm_f2;

        this->ionTerm_f1 = nullptr;
    }
    if (nColdTerm != nullptr)
    {
        for (len_t ir = 0; ir < nr; ir++)
            delete[] nColdTerm[ir];
        for (len_t ir = 0; ir < nr + 1; ir++)
            delete[] nColdTerm_fr[ir];
        delete[] nColdTerm;
        delete[] nColdTerm_fr;

        this->nColdTerm = nullptr;
    }
    if (screenedTerm != nullptr)
    {
        delete[] screenedTerm;
        delete[] screenedTerm_fr;

        this->screenedTerm = nullptr;
    }
    if (screenedTerm_f1 != nullptr)
    {
        delete[] screenedTerm_f1;
        delete[] screenedTerm_f2;

        this->screenedTerm_f1 = nullptr;
    }
    if (stoppingTerm != nullptr)
    {
        delete[] stoppingTerm;
        delete[] stoppingTerm_fr;

        this->stoppingTerm = nullptr;
    }
    if (stoppingTerm_f1 != nullptr)
    {
        delete[] stoppingTerm_f1;
        delete[] stoppingTerm_f2;

        this->stoppingTerm_f1 = nullptr;
    }
    if (ionIZ != nullptr)
    {
        delete[] ionIZ;
        delete[] ionZ0;

        this->ionIZ = nullptr;
    }
    if (lnLEE_partialNi != nullptr)
    {
        for (len_t ir = 0; ir < nr; ir++)
        {
            delete[] lnLEE_partialNi[ir];
            delete[] lnLEI_partialNi[ir];
        }
        delete[] lnLEE_partialNi;
        delete[] lnLEI_partialNi;

        this->lnLEE_partialNi = nullptr;
    }
    if (nColdTerm_f1 != nullptr)
    {
        for (len_t ir = 0; ir < nr; ir++)
        {
            delete[] nColdTerm_f1[ir];
            delete[] nColdTerm_f2[ir];
        }
        delete[] nColdTerm_f1;
        delete[] nColdTerm_f2;

        this->nColdTerm_f1 = nullptr;
    }
    if (atomicParameter != nullptr)
    {
        delete[] atomicParameter;

        this->atomicParameter = nullptr;
    }
    if (nonlinearMat != nullptr)
    {
        for (len_t i = 0; i < np1 + 1; i++)
            delete[] nonlinearMat[i];
        delete[] nonlinearMat;
        delete[] trapzWeights;

        this->nonlinearMat = nullptr;
    }
}

/**
 * Initializes a GSL workspace for evaluation of full e-e test particle operator
 */
void CollisionFrequency::InitializeGSLWorkspace()
{
    DeallocateGSL();
    gsl_ad_w = gsl_integration_workspace_alloc(1000);
}

/**
 * Deallocator
 */
void CollisionFrequency::DeallocateGSL()
{
    if (this->gsl_ad_w != nullptr)
    {
        gsl_integration_workspace_free(gsl_ad_w);
        this->gsl_ad_w = nullptr;
    }
}

/**
 * Evaluates the Jacobian with respect to unknown derivId of the
 * collision frequency at radial grid point ir and momentum p.
 */
real_t CollisionFrequency::evaluatePartialAtP(len_t ir, real_t p, len_t derivId, len_t n, struct collqty_settings *inSettings)
{
    // Return 0 for all other derivId but ncold, ni or Tcold (when collfreq mode is FULL)
    if (!(derivId == id_ncold || derivId == id_ni || derivId == id_Tcold))
        return 0;

    bool isPartiallyScreened = (inSettings->collfreq_type == OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_PARTIALLY_SCREENED || inSettings->collfreq_type == OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_PARTIALLY_SCREENED_WALKOWIAK);
    bool isNonScreened = (inSettings->collfreq_type == OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_NON_SCREENED);
    bool isBrems = (inSettings->bremsstrahlung_mode != OptionConstants::EQTERM_BREMSSTRAHLUNG_MODE_NEGLECT);

    real_t preFact = evaluatePreFactorAtP(p, inSettings->collfreq_mode);
    real_t lnLee = lnLambdaEE->evaluateAtP(ir, p, inSettings);
    real_t lnLei = lnLambdaEI->evaluateAtP(ir, p, inSettings);

    real_t electronTerm = evaluateElectronTermAtP(ir, p, inSettings->collfreq_mode);

    // if ncold, this is the jacobian
    if (derivId == id_ncold)
        return preFact * lnLee * electronTerm;
    // else, for ions or Tcold, we move on...

    real_t dLnLee = lnLambdaEE->evaluatePartialAtP(ir, p, derivId, n, inSettings);
    real_t dLnLei = lnLambdaEI->evaluatePartialAtP(ir, p, derivId, n, inSettings);

    real_t ntarget = GetNTarget(ir, isNonScreened);
    // evaluate and return T_cold expression
    if (derivId == id_Tcold)
    {
        real_t DDTelectronTerm = lnLee * evaluateDDTElectronTermAtP(ir, p, inSettings->collfreq_mode) + dLnLee * electronTerm;
        real_t electronContribution = preFact * ntarget * DDTelectronTerm;
        real_t ionContribution = 0;
        if (hasIonTerm)
            for (len_t iz = 0; iz < nZ; iz++)
                for (len_t Z0 = 0; Z0 <= Zs[iz]; Z0++)
                {
                    len_t Zfact = Z0 * Z0;
                    len_t indZ = ionIndex[iz][Z0];
                    if (isNonScreened)
                        Zfact = Zs[iz] * Zs[iz];
                    ionContribution += preFact * dLnLei * Zfact * evaluateIonTermAtP(iz, Z0, p) * ionDensities[ir][indZ];
                }
        return electronContribution + ionContribution;
    }

    // else treat n_i case
    // set iz and Z0 corresponding to input nMultiple "n"
    len_t iz_in, Z0_in;
    ionHandler->GetIonIndices(n, iz_in, Z0_in);
    len_t Zs_in = Zs[iz_in];

    real_t collFreq = dLnLee * electronTerm * ntarget;
    if (isNonScreened)
        collFreq += (Zs_in - Z0_in) * lnLee * electronTerm;

    // Add ion contribution; SlowingDownFrequency doesn't have one and will skip this step
    if (hasIonTerm)
    {
        len_t Zfact;
        if (isNonScreened)
            Zfact = Zs_in * Zs_in;
        else
            Zfact = Z0_in * Z0_in;
        collFreq += (lnLei + dLnLei * ionDensities[ir][n]) * Zfact * evaluateIonTermAtP(iz_in, Z0_in, p);
    }
    // Add bound-electron contribution
    if (isPartiallyScreened)
        collFreq += evaluateScreenedTermAtP(iz_in, Z0_in, p, inSettings->collfreq_type);
    if (isPartiallyScreened)
        collFreq += evaluateStoppingTermAtP(iz_in, Z0_in, p, inSettings->collfreq_mode);

    collFreq *= preFact;

    // Add Bremsstrahlung contribution
    if (isBrems)
        collFreq += evaluateBremsstrahlungTermAtP(iz_in, Z0_in, p, inSettings->bremsstrahlung_mode);

    return collFreq;
}
