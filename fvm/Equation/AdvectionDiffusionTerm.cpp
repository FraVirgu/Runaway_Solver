/**
 * Implementation of a combined advection-diffusion term.
 */

#include <algorithm>
#include "FVM/Equation/AdvectionDiffusionTerm.hpp"
#include "FVM/Equation/TermLog.hpp"


using namespace DREAM::FVM;
using namespace std;

/**
 * Destructor.
 */
AdvectionDiffusionTerm::~AdvectionDiffusionTerm(){
    for (auto dif : this->diffusionterms)
        delete dif;
    for (auto adv : this->advectionterms)
        delete adv;
}

/**
 * Add an advection term to this term.
 */
void AdvectionDiffusionTerm::Add(AdvectionTerm *a) {
    a->SetCoefficients(this->fr, this->f1, this->f2, this->f1pSqAtZero, this->AdvectionTerm::deltaRadialFlux);
    a->SetInterpolationCoefficients(this->deltar, this->delta1, this->delta2);
    advectionterms.push_back(a);
}

/**
 * Add a diffusion term to this term.
 */
void AdvectionDiffusionTerm::Add(DiffusionTerm *d) {
    d->SetCoefficients(
        this->drr, this->d11, this->d12, this->d21, this->d22, this->DiffusionTerm::deltaRadialFlux
    );
    diffusionterms.push_back(d);
}

/**
 * Returns the number of non-zero elements per row
 * inserted by this term into a linear operator matrix.
 */
len_t AdvectionDiffusionTerm::GetNumberOfNonZerosPerRow() const {
    len_t nnz = 0;
    for (auto it = advectionterms.begin(); it != advectionterms.end(); it++)
        nnz = max(nnz, (*it)->GetNumberOfNonZerosPerRow());

    for (auto it = diffusionterms.begin(); it != diffusionterms.end(); it++)
        nnz = max(nnz, (*it)->GetNumberOfNonZerosPerRow());

    return nnz;
}

/**
 * Helper function for `GetNumberOfNonZerosPerRow_jac()` which extracts
 * the number of unique nMultiples contributing to the jacobian of the 
 * EquationTerm `term` that are not already accounted for in inVec 
 * 
 *   term: an equation term owned by this AdvectionDiffusionTerm 
 *  inVec: a vector tracking all derivIds in the AdvectionDiffusionTerm
 *         that contribute to the jacobian block of this Operator
 */
len_t NumberOfOffdiagonalJacobianNNZToAdd(EquationTerm *term, std::vector<len_t>&inVec) {
    std::vector<len_t> derivIds   = term->GetDerivIdsJacobian(); // off-diagonal derivIds contributing to term 
    std::vector<len_t> nMultiples = term->GetNMultiplesJacobian(); // corresponding nMultiples of the unknown
    len_t nMultToAdd = 0;
    // sum over all derivIds that contribute to the `term` jacobian
    for(len_t i=0; i<derivIds.size(); i++) 
        // check if inVec does not already contain the derivId 
        if(std::find(inVec.begin(), inVec.end(), derivIds[i]) == inVec.end()){
            inVec.push_back(derivIds[i]); // otherwise add it
            nMultToAdd += nMultiples[i];  // and add to length of nMultToAdd
        }
    return nMultToAdd;
}

/**
 * Returns the number of non-zero elements per row
 * inserted by this term into a jacobian matrix.
 */
len_t AdvectionDiffusionTerm::GetNumberOfNonZerosPerRow_jac() const {
    // first add contributions from the `diagonal` jacobian block 
    len_t nnz = this->GetNumberOfNonZerosPerRow(); 

    // next add off-diagonal contributions to the jacobian:
    std::vector<len_t> derivIdsInJacobian;
    len_t nOffdiagonalElements = 0;
    for (auto it = advectionterms.begin(); it != advectionterms.end(); it++)
        nOffdiagonalElements += NumberOfOffdiagonalJacobianNNZToAdd(*it, derivIdsInJacobian);
    for (auto it = diffusionterms.begin(); it != diffusionterms.end(); it++)
        nOffdiagonalElements += NumberOfOffdiagonalJacobianNNZToAdd(*it, derivIdsInJacobian);

    return nnz + nOffdiagonalElements;
}

/**
 * Reset the advection and diffusion coefficients of this term.
 */
void AdvectionDiffusionTerm::ResetCoefficients() {
    if (this->advectionterms.size() > 0)
        this->AdvectionTerm::ResetCoefficients();
    if (this->diffusionterms.size() > 0)
        this->DiffusionTerm::ResetCoefficients();
}

/**
 * Rebuild this equation term.
 *
 * t: Simulation time to rebuild term for.
 */
void AdvectionDiffusionTerm::Rebuild(const real_t t, const real_t dt, UnknownQuantityHandler *uqty) {
    const bool log = TermLog::rebuild();
    TermLog::clk::time_point t0 = TermLog::clk::now();

    this->ResetCoefficients();
    if (log)
        printf("[rebuild]                 reset coefficients %38s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));

    // Rebuild advection-diffusion coefficients
    for (auto it = advectionterms.begin(); it != advectionterms.end(); it++){
        t0 = TermLog::clk::now();
        (*it)->Rebuild(t, dt, uqty);
        if (log)
            printf("[rebuild]                 advection  %-40s %9.3f ms\n",
                   TermLog::Label(*it).c_str(), TermLog::ms(t0, TermLog::clk::now()));
    }

    for (auto it = diffusionterms.begin(); it != diffusionterms.end(); it++){
        t0 = TermLog::clk::now();
        (*it)->Rebuild(t, dt, uqty);
        if (log)
            printf("[rebuild]                 diffusion  %-40s %9.3f ms\n",
                   TermLog::Label(*it).c_str(), TermLog::ms(t0, TermLog::clk::now()));
    }

    t0 = TermLog::clk::now();
    this->AdvectionTerm::RebuildFluxLimiterDamping(t, dt);
    if (log)
        printf("[rebuild]                 flux-limiter damping %36s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    t0 = TermLog::clk::now();
    this->AdvectionTerm::RebuildInterpolationCoefficients(uqty, this->drr, this->d11, this->d22);
    if (log)
        printf("[rebuild]                 advection interpolation coefficients (upwind) %17s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
}

/**
 * Insert the combined advection and diffusion terms into the matrix.
 *
 * The coefficients of all advection terms have already been summed into
 * F1, F2 (and those of the diffusion terms into D11, D22), so the matrix
 * is filled in one pass for the advection and one for the diffusion. The
 * log lists which terms went into each pass.
 */
void AdvectionDiffusionTerm::SetMatrixElements(Matrix *mat, real_t *rhs) {
    const bool log = TermLog::matrix();
    TermLog::clk::time_point t0;

    if (this->advectionterms.size() > 0) {
        if (log) {
            printf("[matrix]                insert advection (sum of:");
            for (auto it = advectionterms.begin(); it != advectionterms.end(); it++)
                printf(" %s;", TermLog::Label(*it).c_str());
            printf(")\n");
        }
        t0 = TermLog::clk::now();
        this->AdvectionTerm::SetMatrixElements(mat, rhs);
        if (log)
            printf("[matrix]                  advection inserted %38s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    }
    if (this->diffusionterms.size() > 0) {
        if (log) {
            printf("[matrix]                insert diffusion (sum of:");
            for (auto it = diffusionterms.begin(); it != diffusionterms.end(); it++)
                printf(" %s;", TermLog::Label(*it).c_str());
            printf(")\n");
        }
        t0 = TermLog::clk::now();
        this->DiffusionTerm::SetMatrixElements(mat, rhs);
        if (log)
            printf("[matrix]                  diffusion inserted %38s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    }
}

/**
 * Add the combined advection and diffusion terms to the right-hand side
 * (same structure as SetMatrixElements()).
 */
void AdvectionDiffusionTerm::SetVectorElements(real_t *vec, const real_t *x) {
    const bool log = TermLog::matrix();
    TermLog::clk::time_point t0;

    if (this->advectionterms.size() > 0) {
        t0 = TermLog::clk::now();
        this->AdvectionTerm::SetVectorElements(vec, x);
        if (log)
            printf("[matrix]                  advection added to RHS %34s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    }
    if (this->diffusionterms.size() > 0) {
        t0 = TermLog::clk::now();
        this->DiffusionTerm::SetVectorElements(vec, x);
        if (log)
            printf("[matrix]                  diffusion added to RHS %34s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    }
}

/**
 * Save the advection and diffusion coefficients of this
 * object to the specified file.
 */
void AdvectionDiffusionTerm::SaveCoefficientsSFile(const string& filename) {
    SFile *sf = SFile::Create(filename, SFILE_MODE_WRITE);
    this->SaveCoefficientsSFile(sf);
    sf->Close();
}
void AdvectionDiffusionTerm::SaveCoefficientsSFile(SFile *sf) {
    this->AdvectionTerm::SaveCoefficientsSFile(sf);
    this->DiffusionTerm::SaveCoefficientsSFile(sf);
}

