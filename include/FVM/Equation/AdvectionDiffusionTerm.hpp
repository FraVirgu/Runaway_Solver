#ifndef _DREAM_FVM_ADVECTION_DIFFUSION_TERM_HPP
#define _DREAM_FVM_ADVECTION_DIFFUSION_TERM_HPP

#include <vector>
#include "FVM/Equation/AdvectionTerm.hpp"
#include "FVM/Equation/DiffusionTerm.hpp"
#include "FVM/Equation/EquationTerm.hpp"
#include "FVM/UnknownQuantityHandler.hpp"

namespace DREAM::FVM
{
    class AdvectionDiffusionTerm : public AdvectionTerm, public DiffusionTerm
    {

    private:
        std::vector<AdvectionTerm *> advectionterms;
        std::vector<DiffusionTerm *> diffusionterms;

        // F1 and D11 on the upper p boundary (last p1 face) of every xi line
        // at every radius, [offset(ir) + j], complete on every rank (see
        // GatherUpperBoundaryCoefficients())
        std::vector<real_t> upperF1, upperD11;
        std::vector<len_t> upperOffset;

        // Coefficients on the faces read by the trapping boundary condition
        // (PXiInternalTrapping), complete on every rank (see
        // GatherTrappingBoundaryCoefficients()). Empty without trapping.
        //   trapJm[ir]      first negative trapped xi cell (face jm = -xi_T);
        //                   n2[ir] if there is none at this radius
        //   trapKm[ir]      xi lines with a negative trapped radial face ir
        //   trapXi          per (ir, i):     [F2, D22, D21, deltas, deltas_jac]
        //   trapR           per (ir, q, i):  [Fr, Drr, deltas, deltas_jac], km = trapKm[ir][q]
        std::vector<len_t> trapJm;
        std::vector<std::vector<len_t>> trapKm;
        std::vector<len_t> trapXiOffset, trapROffset;
        std::vector<real_t> trapXi, trapR;
        len_t trapW = 0, trapXiStride = 0, trapRStride = 0;

    public:
        // Positions in the slots returned by GetTrappingXi() and GetTrappingR()
        static constexpr len_t TRAP_F2 = 0, TRAP_D22 = 1, TRAP_D21 = 2, TRAP_XI_DELTA = 3;
        static constexpr len_t TRAP_FR = 0, TRAP_DRR = 1, TRAP_R_DELTA = 2;

        // The coefficients themselves are allocated by the first Rebuild(),
        // for the cells owned by this rank (see UpdateCoefficientLayout())
        AdvectionDiffusionTerm(Grid *g)
            : AdvectionTerm(g, false), DiffusionTerm(g, false)
        {
            this->AdvectionTerm::AllocateCoefficientTables();
            this->DiffusionTerm::AllocateCoefficientTables();
            this->AdvectionTerm::AllocateInterpolationCoefficients();
        }
        ~AdvectionDiffusionTerm();

        // F1 and D11 on the upper p boundary of xi line j at radius ir (on
        // every rank, once GatherUpperBoundaryCoefficients() has been called)
        real_t GetUpperBoundaryF1(const len_t ir, const len_t j) const { return upperF1[upperOffset[ir] + j]; }
        real_t GetUpperBoundaryD11(const len_t ir, const len_t j) const { return upperD11[upperOffset[ir] + j]; }

        void Add(AdvectionTerm *);
        void Add(DiffusionTerm *);

        virtual len_t GetNumberOfNonZerosPerRow() const override;
        virtual len_t GetNumberOfNonZerosPerRow_jac() const override;

        virtual void Rebuild(const real_t, const real_t, UnknownQuantityHandler *) override;
        virtual void ResetCoefficients() override;
        void GatherUpperBoundaryCoefficients();
        void GatherTrappingBoundaryCoefficients();

        // Coefficients of the trapping boundary condition (on every rank, once
        // GatherTrappingBoundaryCoefficients() has been called)
        bool HasTrappingCoefficients() const { return !trapJm.empty(); }
        len_t GetTrappingJm(const len_t ir) const { return trapJm[ir]; }
        const std::vector<len_t> &GetTrappingKm(const len_t ir) const { return trapKm[ir]; }
        // Face jm, momentum index i: [TRAP_F2], [TRAP_D22], [TRAP_D21]
        const real_t *GetTrappingXi(const len_t ir, const len_t i) const
        { return trapXi.data() + trapXiOffset[ir] + i * trapXiStride; }
        const real_t *GetTrappingXiDelta(const len_t ir, const len_t i, const bool jacobian) const
        { return GetTrappingXi(ir, i) + TRAP_XI_DELTA + (jacobian ? trapW : 0); }
        // Radial face ir, line km = GetTrappingKm(ir)[q], momentum index i: [TRAP_FR], [TRAP_DRR]
        const real_t *GetTrappingR(const len_t ir, const len_t q, const len_t i) const
        { return trapR.data() + trapROffset[ir] + (q * GetTrappingNp(ir) + i) * trapRStride; }
        const real_t *GetTrappingRDelta(const len_t ir, const len_t q, const len_t i, const bool jacobian) const
        { return GetTrappingR(ir, q, i) + TRAP_R_DELTA + (jacobian ? trapW : 0); }
        len_t GetTrappingNp(const len_t ir) const { return this->AdvectionTerm::n1[ir]; }
        void PrintCoefficientNorms(const real_t dt);

        const std::vector<AdvectionTerm *> &GetAdvectionTerms() const { return advectionterms; }
        const std::vector<DiffusionTerm *> &GetDiffusionTerms() const { return diffusionterms; }

        virtual const real_t *GetRadialJacobianInterpolationCoeffs() const override
        {
            if (this->advectionterms.size() > 0)
                return this->advectionterms[0]->GetRadialJacobianInterpolationCoeffs();
            else if (this->diffusionterms.size() > 0)
                return this->diffusionterms[0]->GetRadialJacobianInterpolationCoeffs();
            else
                return nullptr;
        }

        virtual bool SetJacobianBlock(
            const len_t uqtyId, const len_t derivId, Matrix *jac, const real_t *x) override { return this->SetJacobianBlock(uqtyId, derivId, jac, x, false); }

        virtual bool SetJacobianBlock(
            const len_t uqtyId, const len_t derivId, Matrix *jac, const real_t *x,
            bool
#ifndef NDEBUG
                printTerms
#endif
        )
        {
            bool contributes = false;

            this->interp_mode = AdvectionInterpolationCoefficient::AD_INTERP_MODE_JACOBIAN;
            // Set diagonal block (assuming constant coefficients)
            if (uqtyId == derivId)
            {
                contributes = true;

                if (this->advectionterms.size() > 0)
                    this->AdvectionTerm::SetMatrixElements(jac, nullptr);
                if (this->diffusionterms.size() > 0)
                    this->DiffusionTerm::SetMatrixElements(jac, nullptr);
            }

            // Handle any off-diagonal blocks and/or non-linear coefficients
            for (auto it = advectionterms.begin(); it != advectionterms.end(); it++)
            {
                bool c = (*it)->SetJacobianBlock(uqtyId, derivId, jac, x);
                contributes |= c;
#ifndef NDEBUG
                if (c && printTerms)
                    printf("Contribution from %s", (*it)->GetName().c_str());
#endif
            }

            for (auto it = diffusionterms.begin(); it != diffusionterms.end(); it++)
            {
                bool c = (*it)->SetJacobianBlock(uqtyId, derivId, jac, x);
                contributes |= c;
#ifndef NDEBUG
                if (c && printTerms)
                    printf("Contribution from %s", (*it)->GetName().c_str());
#endif
            }

            return contributes;
        }
        // Defined in AdvectionDiffusionTerm.cpp (with the optional term-level log)
        virtual void SetMatrixElements(Matrix *mat, real_t *rhs) override;
        virtual void SetVectorElements(real_t *vec, const real_t *x) override;

        virtual void SaveCoefficientsSFile(const std::string &) override;
        virtual void SaveCoefficientsSFile(SFile *) override;
    };
}

#endif /*_DREAM_FVM_ADVECTION_DIFFUSION_TERM_HPP*/
