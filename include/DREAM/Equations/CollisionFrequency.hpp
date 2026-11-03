#ifndef _DREAM_EQUATIONS_COLLISION_FREQUENCY_HPP
#define _DREAM_EQUATIONS_COLLISION_FREQUENCY_HPP

#include "CollisionQuantity.hpp"
#include "DREAM/Equations/CoulombLogarithm.hpp"

namespace DREAM {
    class CollisionFrequency : public CollisionQuantity {
    private:
        void InitializeGSLWorkspace();
        void DeallocateGSL();

        // Stored factors of the collision frequency on one flux grid, from
        // which the partial contributions are evaluated on the fly
        struct gridTerms {
            real_t *const* nColdTerm;
            const real_t *preFactor, *ionTerm, *screenedTerm, *stoppingTerm, *bremsTerm;
            real_t *const* lnLee;
            real_t *const* lnLei;
            len_t n1, n2;   // size of the flux grid
            len_t zStride;  // stride between charge states in the momentum-dependent factors
        };
        gridTerms GetGridTerms(FVM::fluxGridType) const;
        real_t evaluateNColdPartial(const gridTerms&, len_t ir, len_t pind, len_t pindStore) const;
        real_t evaluateNiPartial(const gridTerms&, len_t ir, len_t pind, len_t pindStore, len_t indZ, real_t ntarget, real_t &lnLContrib) const;

    protected:
        bool hasIonTerm;

        real_t **nonlinearMat = nullptr;
        real_t *trapzWeights  = nullptr;
        
        CoulombLogarithm *lnLambdaEE;
        CoulombLogarithm *lnLambdaEI;
        
        real_t **ionDensities = nullptr;
        real_t *Zs            = nullptr;
        real_t **ionIndex     = nullptr;
        len_t *ionIZ          = nullptr; // ion species of charge state indZ
        len_t *ionZ0          = nullptr; // charge of charge state indZ

        // Derivatives of lnLambda with respect to the ion densities,
        // allocated on the radii owned by this rank (size nzs)
        real_t **lnLEE_partialNi = nullptr;
        real_t **lnLEI_partialNi = nullptr;
        void SetLnLambdaPartialNi();

        real_t *preFactor     = nullptr;
        real_t *preFactor_fr  = nullptr;
        real_t *preFactor_f1  = nullptr;
        real_t *preFactor_f2  = nullptr;
        virtual real_t evaluatePreFactorAtP(real_t p, OptionConstants::collqty_collfreq_mode collfreq_mode) = 0; 

        real_t *screenedTerm        = nullptr;
        real_t *screenedTerm_fr     = nullptr;
        real_t *screenedTerm_f1     = nullptr;
        real_t *screenedTerm_f2     = nullptr;
        virtual real_t evaluateScreenedTermAtP(len_t iz, len_t Z0, real_t p,OptionConstants::collqty_collfreq_type collfreq_type) = 0;
        
        real_t *stoppingTerm        = nullptr;
        real_t *stoppingTerm_fr     = nullptr;
        real_t *stoppingTerm_f1     = nullptr;
        real_t *stoppingTerm_f2     = nullptr;
        virtual real_t evaluateStoppingTermAtP(len_t iz, len_t Z0, real_t p, OptionConstants::collqty_collfreq_mode collfreq_mode) = 0;
        
        real_t *ionTerm = nullptr;
        real_t *ionTerm_fr = nullptr;
        real_t *ionTerm_f1 = nullptr;
        real_t *ionTerm_f2 = nullptr;
        virtual real_t evaluateIonTermAtP(len_t iz, len_t Z0, real_t p) = 0;

        real_t **nColdTerm    = nullptr;
        real_t **nColdTerm_fr = nullptr;
        real_t **nColdTerm_f1 = nullptr;
        real_t **nColdTerm_f2 = nullptr;
        virtual real_t evaluateElectronTermAtP(len_t ir, real_t p, OptionConstants::collqty_collfreq_mode collfreq_mode) = 0;
        virtual real_t evaluateDDTElectronTermAtP(len_t ir, real_t p, OptionConstants::collqty_collfreq_mode collfreq_mode) = 0;
        
        real_t *bremsTerm    = nullptr;
        real_t *bremsTerm_fr = nullptr;
        real_t *bremsTerm_f1 = nullptr;
        real_t *bremsTerm_f2 = nullptr;
        virtual real_t evaluateBremsstrahlungTermAtP(len_t iz, len_t Z0, real_t p, OptionConstants::eqterm_bremsstrahlung_mode brems_mode) = 0;

        real_t *atomicParameter = nullptr; // size nzs. Constant term

        // Scaled modified Bessel functions of the first kind
        real_t *K0Scaled = nullptr; // exp(1/Theta)*K0(1/Theta) 
        real_t *K1Scaled = nullptr; // exp(1/Theta)*K1(1/Theta)
        real_t *K2Scaled = nullptr; // exp(1/Theta)*K2(1/Theta)
        
        static real_t psi0Integrand(real_t s, void *params);
        static real_t psi1Integrand(real_t s, void *params);
        static real_t psi2Integrand(real_t s, void *params);
        virtual real_t evaluatePsi0(len_t ir, real_t p);
        virtual real_t evaluatePsi1(len_t ir, real_t p);
        virtual real_t evaluatePsi2(len_t ir, real_t p);
        virtual real_t evaluatePsiLowenergyLimit(len_t n, real_t p, real_t Theta);
        virtual real_t evaluateExp1OverThetaK(real_t Theta, real_t n);
        
        gsl_integration_fixed_workspace **gsl_w = nullptr;
        gsl_integration_workspace *gsl_ad_w = nullptr;
        int QAG_KEY = GSL_INTEG_GAUSS31;

        void setPreFactor(real_t *&preFactor, const real_t *pIn, len_t np1, len_t np2);
        void setElectronTerm(real_t **&nColdTerm, const real_t *pIn, len_t nr, len_t np1, len_t np2);
        void setScreenedTerm(real_t *&screenedTerm, const real_t *pIn, len_t np1, len_t np2);
        void setStoppingTerm(real_t *&stoppingTerm, const real_t *pIn, len_t np1, len_t np2);
        void setIonTerm(real_t *&ionTerm, const real_t *pIn, len_t np1, len_t np2);
        void setBremsTerm(real_t *&bremsTerm, const real_t *pIn, len_t np1, len_t np2);

        virtual void RebuildPlasmaDependentTerms() override;
        virtual void RebuildConstantTerms() override;
        virtual real_t GetAtomicParameter(len_t iz, len_t Z0) = 0;

        virtual void calculateIsotropicNonlinearOperatorMatrix() = 0;
        
        virtual void AllocatePartialQuantities() override;
        void DeallocatePartialQuantities();

        void AllocateRadialQuantities();
        void DeallocateRadialQuantities();

        virtual void AssembleQuantity(real_t **&collisionQuantity, len_t nr, len_t np1, len_t np2, FVM::fluxGridType) override;

        real_t GetNTarget(len_t ir, bool isNonScreened);
    public:
        CollisionFrequency(FVM::Grid *g, FVM::UnknownQuantityHandler *u, IonHandler *ih,  
                CoulombLogarithm *lnLee,CoulombLogarithm *lnLei,
                enum OptionConstants::momentumgrid_type mgtype,  struct collqty_settings *cqset);
        virtual ~CollisionFrequency();

        void RebuildRadialTerms();

        real_t GetUnknownPartialContribution(len_t id_unknown, FVM::fluxGridType, len_t ir, len_t i, len_t j, len_t n);
        virtual real_t evaluateAtP(len_t ir, real_t p, struct collqty_settings *inSettings) override;
        virtual real_t evaluatePartialAtP(len_t ir, real_t p, len_t derivId, len_t nMultiple, struct collqty_settings *inSettings) override;
        using CollisionQuantity::evaluateAtP;
        using CollisionQuantity::evaluatePartialAtP;
    };
}

#endif/*_DREAM_EQUATIONS_COLLISION_FREQUENCY_HPP*/
