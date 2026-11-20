/**
 * Implementation of a combined advection-diffusion term.
 */

#include <algorithm>
#include <cmath>
#include <petscsys.h>
#include "FVM/Equation/AdvectionDiffusionTerm.hpp"
#include "FVM/Equation/TermLog.hpp"

using namespace DREAM::FVM;
using namespace std;

/**
 * Destructor.
 */
AdvectionDiffusionTerm::~AdvectionDiffusionTerm()
{
    for (auto dif : this->diffusionterms)
        delete dif;
    for (auto adv : this->advectionterms)
        delete adv;
}

/**
 * Add an advection term to this term.
 */
void AdvectionDiffusionTerm::Add(AdvectionTerm *a)
{
    a->SetCoefficients(this->fr, this->f1, this->f2, this->f1pSqAtZero, this->AdvectionTerm::deltaRadialFlux);
    a->SetInterpolationCoefficients(this->deltar, this->delta1, this->delta2);
    advectionterms.push_back(a);
}

/**
 * Add a diffusion term to this term.
 */
void AdvectionDiffusionTerm::Add(DiffusionTerm *d)
{
    d->SetCoefficients(
        this->drr, this->d11, this->d12, this->d21, this->d22, this->DiffusionTerm::deltaRadialFlux);
    diffusionterms.push_back(d);
}

/**
 * Returns the number of non-zero elements per row
 * inserted by this term into a linear operator matrix.
 */
len_t AdvectionDiffusionTerm::GetNumberOfNonZerosPerRow() const
{
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
len_t NumberOfOffdiagonalJacobianNNZToAdd(EquationTerm *term, std::vector<len_t> &inVec)
{
    std::vector<len_t> derivIds = term->GetDerivIdsJacobian();     // off-diagonal derivIds contributing to term
    std::vector<len_t> nMultiples = term->GetNMultiplesJacobian(); // corresponding nMultiples of the unknown
    len_t nMultToAdd = 0;
    // sum over all derivIds that contribute to the `term` jacobian
    for (len_t i = 0; i < derivIds.size(); i++)
        // check if inVec does not already contain the derivId
        if (std::find(inVec.begin(), inVec.end(), derivIds[i]) == inVec.end())
        {
            inVec.push_back(derivIds[i]); // otherwise add it
            nMultToAdd += nMultiples[i];  // and add to length of nMultToAdd
        }
    return nMultToAdd;
}

/**
 * Returns the number of non-zero elements per row
 * inserted by this term into a jacobian matrix.
 */
len_t AdvectionDiffusionTerm::GetNumberOfNonZerosPerRow_jac() const
{
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
 *
 * Only the coefficients on the faces of the cells owned by this rank
 * are reset (see 'EquationTerm::GetLocalRange()'), which are the ones
 * the advection and diffusion terms rebuild.
 */
void AdvectionDiffusionTerm::ResetCoefficients()
{
    const bool adv = (this->advectionterms.size() > 0);
    const bool diff = (this->diffusionterms.size() > 0);
    const len_t nr = this->AdvectionTerm::nr;

    for (len_t ir = 0; ir < nr; ir++)
    {
        // Cells of this radius owned by this rank
        len_t i1lo, i1hi, i2lo, i2hi;
        if (!this->AdvectionTerm::GetLocalRange(ir, i1lo, i1hi, i2lo, i2hi))
            continue;

        const len_t np1 = this->AdvectionTerm::n1[ir];

        // Radial faces (inner and outer face of the cells)
        // XXX here we assume that all momentum grids are the same
        for (len_t k = ir; k < ir + 2; k++)
            for (len_t j = i2lo; j < i2hi; j++)
                for (len_t i = i1lo; i < i1hi; i++)
                {
                    if (adv)
                        this->fr[k][j * np1 + i] = 0;
                    if (diff)
                        this->drr[k][j * np1 + i] = 0;
                }

        // Faces in the first momentum coordinate
        for (len_t j = i2lo; j < i2hi; j++)
        {
            for (len_t i = i1lo; i < i1hi + 1; i++)
            {
                if (adv)
                    this->f1[ir][j * (np1 + 1) + i] = 0;
                if (diff)
                {
                    this->d11[ir][j * (np1 + 1) + i] = 0;
                    this->d12[ir][j * (np1 + 1) + i] = 0;
                }
            }
            if (adv)
                this->f1pSqAtZero[ir][j] = 0;
        }

        // Faces in the second momentum coordinate
        for (len_t j = i2lo; j < i2hi + 1; j++)
            for (len_t i = i1lo; i < i1hi; i++)
            {
                if (adv)
                    this->f2[ir][j * np1 + i] = 0;
                if (diff)
                {
                    this->d22[ir][j * np1 + i] = 0;
                    this->d21[ir][j * np1 + i] = 0;
                }
            }
    }
}

/**
 * Rebuild this equation term.
 *
 * t: Simulation time to rebuild term for.
 */
void AdvectionDiffusionTerm::Rebuild(const real_t t, const real_t dt, UnknownQuantityHandler *uqty)
{
    const bool log = TermLog::rebuild();
    TermLog::clk::time_point t0 = TermLog::clk::now();

    // Allocate the coefficients for the cells owned by this rank (once the
    // matrix rows of this rank are known, and again if they change)
    this->AdvectionTerm::UpdateCoefficientLayout();
    this->DiffusionTerm::UpdateCoefficientLayout();

    this->ResetCoefficients();
    if (log)
        printf("[rebuild]                 reset coefficients %38s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));

    // Rebuild advection-diffusion coefficients
    for (auto it = advectionterms.begin(); it != advectionterms.end(); it++)
    {
        t0 = TermLog::clk::now();
        (*it)->Rebuild(t, dt, uqty);
        if (log)
            printf("[rebuild]                 advection  %-40s %9.3f ms\n",
                   TermLog::Label(*it).c_str(), TermLog::ms(t0, TermLog::clk::now()));
    }

    for (auto it = diffusionterms.begin(); it != diffusionterms.end(); it++)
    {
        t0 = TermLog::clk::now();
        (*it)->Rebuild(t, dt, uqty);
        if (log)
            printf("[rebuild]                 diffusion  %-40s %9.3f ms\n",
                   TermLog::Label(*it).c_str(), TermLog::ms(t0, TermLog::clk::now()));
    }

    t0 = TermLog::clk::now();
    // this->AdvectionTerm::RebuildFluxLimiterDamping(t, dt);
    if (log)
        printf("[rebuild]      SKIP           flux-limiter damping %36s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    t0 = TermLog::clk::now();
    this->AdvectionTerm::RebuildInterpolationCoefficients(uqty, this->drr, this->d11, this->d22);
    if (log)
        printf("[rebuild]            advection interpolation coefficients (upwind) %17s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));

    if (log)
        this->PrintCoefficientNorms(dt);
}

/**
 * Print how strongly each row is coupled in r, p and xi relative to its
 * diagonal. For every cell, with h the cell width in that direction and
 * each coefficient the mean of |coeff| on the two faces of the cell,
 *
 *   c_r  = dt (|A_r|/dr  + D_rr/dr^2)
 *   c_p  = dt (|A_p|/dp  + D_pp/dp^2   + D_pxi/(dp dxi))
 *   c_xi = dt (|A_xi|/dxi + D_xixi/dxi^2 + D_xip/(dp dxi))
 *
 *   theta_r = c_r / (1 + c_r + c_p + c_xi)
 *
 * c_* are dimensionless (rate x dt). theta_r is the fraction of the row
 * that couples to the neighbouring radii, i.e. what block Jacobi over the
 * radius drops; it is a scale estimate (metric factors V' are left out).
 * Printed: the mean of c_r, c_p, c_xi and of theta_r over the N cells of
 * the unknown (all ranks), and the max of theta_r.
 *
 * Collective: must be called by all ranks.
 */
void AdvectionDiffusionTerm::PrintCoefficientNorms(const real_t dt)
{
    const bool adv = (this->advectionterms.size() > 0);
    const bool diff = (this->diffusionterms.size() > 0);
    const len_t nr = this->AdvectionTerm::nr;
    const len_t *n1 = this->AdvectionTerm::n1;
    const len_t *n2 = this->AdvectionTerm::n2;
    Grid *grid = this->AdvectionTerm::grid;

    // [c_r, c_p, c_xi, theta_r]
    real_t s[4] = {0, 0, 0, 0}, thetaMax = 0;
    len_t N = 0, offset = 0;
    for (len_t ir = 0; ir < nr; ir++)
    {
        const len_t np1 = n1[ir], np2 = n2[ir];
        const real_t dr = grid->GetRadialGrid()->GetDr(ir);
        const real_t *dp1 = grid->GetMomentumGrid(ir)->GetDp1();
        const real_t *dp2 = grid->GetMomentumGrid(ir)->GetDp2();
        N += np1 * np2;
        for (len_t j = 0; j < np2; j++)
            for (len_t i = 0; i < np1; i++)
            {
                if (!this->AdvectionTerm::IsLocalRow(offset + j * np1 + i))
                    continue;

                const len_t c = j * np1 + i;        // cell / lower radial face / lower xi face
                const len_t f1 = j * (np1 + 1) + i; // lower p face
                const real_t h1 = dp1[i], h2 = dp2[j];
                // mean of |x| on the lower and upper face of the cell
                auto m = [](const real_t lo, const real_t hi)
                { return 0.5 * (fabs(lo) + fabs(hi)); };

                real_t cr = 0, cp = 0, cx = 0;
                if (adv)
                {
                    cr += m(this->fr[ir][c], this->fr[ir + 1][c]) / dr;
                    cp += m(this->f1[ir][f1], this->f1[ir][f1 + 1]) / h1;
                    cx += m(this->f2[ir][c], this->f2[ir][c + np1]) / h2;
                }
                if (diff)
                {
                    cr += m(this->drr[ir][c], this->drr[ir + 1][c]) / (dr * dr);
                    cp += m(this->d11[ir][f1], this->d11[ir][f1 + 1]) / (h1 * h1) + m(this->d12[ir][f1], this->d12[ir][f1 + 1]) / (h1 * h2);
                    cx += m(this->d22[ir][c], this->d22[ir][c + np1]) / (h2 * h2) + m(this->d21[ir][c], this->d21[ir][c + np1]) / (h1 * h2);
                }
                cr *= dt;
                cp *= dt;
                cx *= dt;

                const real_t theta = cr / (1 + cr + cp + cx);
                s[0] += cr;
                s[1] += cp;
                s[2] += cx;
                s[3] += theta;
                thetaMax = std::max(thetaMax, theta);
            }
        offset += np1 * np2;
    }

    MPI_Allreduce(MPI_IN_PLACE, s, 4, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &thetaMax, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    for (len_t k = 0; k < 4; k++)
        s[k] /= N;

    printf("[rebuild]                 coupling per row (N = %llu, dt = %.3e): dt*(|A|/h + D/h^2), mean over cells\n",
           (unsigned long long)N, (double)dt);
    printf("[rebuild]                   c_r = %.5e    c_p = %.5e    c_xi = %.5e\n", s[0], s[1], s[2]);
    printf("[rebuild]                   theta_r = c_r / (1 + c_r + c_p + c_xi):  mean %.5e    max %.5e\n",
           s[3], thetaMax);
}

/**
 * Collect the coefficients F1 and D11 on the upper p boundary (the last p1
 * face, i = np1, of every xi line at every radius) on every rank, in
 * 'upperF1' and 'upperD11' (see GetUpperBoundaryF1/D11()).
 *
 * Each rank only builds and stores the coefficients of the cells it owns,
 * but the boundary condition coupling two momentum grids
 * (PXiExternalKineticKinetic) reads the coefficients of the lower grid
 * on this face when it sets the rows of the upper grid, which are in
 * general owned by another rank.
 *
 * Every rank contributes the faces whose cell (i = np1-1) it owns, so
 * that each value comes from exactly one rank, and the sum over all
 * ranks is kept on every rank.
 *
 * Collective: must be called by all ranks.
 */
void AdvectionDiffusionTerm::GatherUpperBoundaryCoefficients()
{
    const len_t nr = this->AdvectionTerm::nr;
    const len_t *n1 = this->AdvectionTerm::n1;
    const len_t *n2 = this->AdvectionTerm::n2;

    const bool adv = (this->advectionterms.size() > 0);
    const bool diff = (this->diffusionterms.size() > 0);

    len_t nFaces = 0;
    upperOffset.resize(nr);
    for (len_t ir = 0; ir < nr; ir++)
    {
        upperOffset[ir] = nFaces;
        nFaces += n2[ir];
    }

    // [F1 of all faces, D11 of all faces]
    std::vector<real_t> buf(2 * nFaces, 0);

    len_t offset = 0, k = 0;
    for (len_t ir = 0; ir < nr; ir++)
    {
        const len_t np1 = n1[ir];
        for (len_t j = 0; j < n2[ir]; j++, k++)
        {
            if (!this->AdvectionTerm::IsLocalRow(offset + j * np1 + np1 - 1))
                continue;

            if (adv)
                buf[k] = this->f1[ir][j * (np1 + 1) + np1];
            if (diff)
                buf[nFaces + k] = this->d11[ir][j * (np1 + 1) + np1];
        }
        offset += np1 * n2[ir];
    }

    PetscMPIInt nRanks;
    MPI_Comm_size(PETSC_COMM_WORLD, &nRanks);
    if (nRanks > 1)
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), (int)(2 * nFaces), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

    upperF1.assign(buf.begin(), buf.begin() + nFaces);
    upperD11.assign(buf.begin() + nFaces, buf.end());
}

/**
 * Collect, on every rank, the coefficients read by the trapping boundary
 * condition (PXiInternalTrapping) when it moves the flux entering the
 * negative trapped region onto the positive side.
 *
 * At every radius ir:
 *   jm  first xi cell of the negative trapped region: its lower xi face
 *       (face jm of the P2 flux grid) is the face at -xi_T. The flux
 *       through it (F2, D22, D21 and the xi interpolation deltas, for every
 *       momentum index i) is added to the rows of the mirrored line jp.
 *   km  every xi line whose cell is negative trapped at the radial face
 *       (Grid::IsNegativePitchTrappedIgnorableRadialFluxCell). The flux
 *       through radial face ir of that line (Fr, Drr and the radial
 *       deltas) is added to the rows of the mirrored line kp.
 *
 * Each rank only stores the coefficients of the cells whose rows it owns,
 * and jm (km) and jp (kp) are in general on different ranks. As in
 * GatherUpperBoundaryCoefficients(), the layout of the buffer follows from
 * the grid only (the same on every rank), every value is filled by exactly
 * one rank (the owner of the cell (ir, jm, i), resp. (ir, km, i), next to
 * the face, which stores and computes it), the others leave 0, and the sum
 * over all ranks is kept on every rank (see GetTrappingXi/R()).
 *
 * Both sets of interpolation deltas are kept (FULL and JACOBIAN mode; they
 * are equal when JacobianIsSame()).
 *
 * Collective: must be called by all ranks, after Rebuild() (the
 * interpolation coefficients must have been built).
 */
void AdvectionDiffusionTerm::GatherTrappingBoundaryCoefficients()
{
    Grid *g = this->AdvectionTerm::grid;
    const len_t nr = this->AdvectionTerm::nr;
    const len_t *n1 = this->AdvectionTerm::n1;
    const len_t *n2 = this->AdvectionTerm::n2;

    trapJm.clear();
    trapKm.clear();
    trapXiOffset.clear();
    trapROffset.clear();
    trapXi.clear();
    trapR.clear();

    // No trapped region, or a fluid grid: nothing to collect. Both are the
    // same on every rank, so all ranks return together.
    if (!g->HasTrapped() || g->GetNCells() == nr)
        return;

    const bool adv = (this->advectionterms.size() > 0);
    const bool diff = (this->diffusionterms.size() > 0);

    AdvectionInterpolationCoefficient *dXi = this->AdvectionTerm::delta2;
    AdvectionInterpolationCoefficient *dR = this->AdvectionTerm::deltar;
    const auto FULL = AdvectionInterpolationCoefficient::AD_INTERP_MODE_FULL;
    const auto JAC = AdvectionInterpolationCoefficient::AD_INTERP_MODE_JACOBIAN;

    // Slot of one face value: [F2, D22, D21, deltas (FULL), deltas (JACOBIAN)]
    // and [Fr, Drr, deltas (FULL), deltas (JACOBIAN)]
    trapW = dXi->GetStencilSize();
    trapXiStride = TRAP_XI_DELTA + 2 * trapW;
    trapRStride = TRAP_R_DELTA + 2 * trapW;

    // (a) Layout, from the (replicated) grid only: the same on every rank
    trapJm.assign(nr, 0);
    trapKm.assign(nr, std::vector<len_t>());
    trapXiOffset.assign(nr + 1, 0);
    trapROffset.assign(nr + 1, 0);
    for (len_t ir = 0; ir < nr; ir++)
    {
        const len_t np = n1[ir], nxi = n2[ir];

        len_t jm = nxi;
        for (len_t j = 0; j < nxi; j++)
            if (g->IsNegativePitchTrappedIgnorableCell(ir, j))
            {
                jm = j;
                break;
            }
        trapJm[ir] = jm;

        for (len_t km = 0; km < nxi; km++)
            if (g->IsNegativePitchTrappedIgnorableRadialFluxCell(ir, km))
                trapKm[ir].push_back(km);

        trapXiOffset[ir + 1] = trapXiOffset[ir] + (jm < nxi ? np * trapXiStride : 0);
        trapROffset[ir + 1] = trapROffset[ir] + trapKm[ir].size() * np * trapRStride;
    }

    // Both parts in one buffer, for a single Allreduce
    const len_t NX = trapXiOffset[nr], NR = trapROffset[nr];
    std::vector<real_t> buf(NX + NR, 0);

    // (b) Fill the values of the cells owned by this rank
    len_t offset = 0;
    for (len_t ir = 0; ir < nr; ir++)
    {
        const len_t np = n1[ir], nxi = n2[ir];
        const len_t jm = trapJm[ir];

        // XI: face jm, filled by the owner of cell (ir, jm, i)
        if (jm < nxi)
        {
            for (len_t i = 0; i < np; i++)
            {
                if (!this->AdvectionTerm::IsLocalRow(offset + jm * np + i))
                    continue;

                real_t *b = buf.data() + trapXiOffset[ir] + i * trapXiStride;
                if (adv)
                {
                    b[TRAP_F2] = this->AdvectionTerm::F2(ir, i, jm, this->AdvectionTerm::f2);
                    if (dXi->IsStored(ir, jm))
                    {
                        const real_t *d = dXi->GetCoefficient(ir, i, jm, FULL);
                        const real_t *dj = dXi->GetCoefficient(ir, i, jm, JAC);
                        for (len_t n = 0; n < trapW; n++)
                        {
                            b[TRAP_XI_DELTA + n] = d[n];
                            b[TRAP_XI_DELTA + trapW + n] = dj[n];
                        }
                    }
                }
                if (diff)
                {
                    b[TRAP_D22] = this->DiffusionTerm::D22(ir, i, jm, this->DiffusionTerm::d22);
                    b[TRAP_D21] = this->DiffusionTerm::D21(ir, i, jm, this->DiffusionTerm::d21);
                }
            }
        }

        // RADIAL: face ir of every line km, filled by the owner of cell (ir, km, i)
        for (len_t q = 0; q < trapKm[ir].size(); q++)
        {
            const len_t km = trapKm[ir][q];
            for (len_t i = 0; i < np; i++)
            {
                if (!this->AdvectionTerm::IsLocalRow(offset + km * np + i))
                    continue;

                real_t *b = buf.data() + NX + trapROffset[ir] + (q * np + i) * trapRStride;
                if (adv)
                {
                    b[TRAP_FR] = this->AdvectionTerm::Fr(ir, i, km, this->AdvectionTerm::fr);
                    if (dR->IsStored(ir, km))
                    {
                        const real_t *d = dR->GetCoefficient(ir, i, km, FULL);
                        const real_t *dj = dR->GetCoefficient(ir, i, km, JAC);
                        for (len_t n = 0; n < trapW; n++)
                        {
                            b[TRAP_R_DELTA + n] = d[n];
                            b[TRAP_R_DELTA + trapW + n] = dj[n];
                        }
                    }
                }
                if (diff)
                    b[TRAP_DRR] = this->DiffusionTerm::Drr(ir, i, km, this->DiffusionTerm::drr);
            }
        }

        offset += np * nxi;
    }

    // (c) Sum over ranks: every value comes from exactly one rank
    PetscMPIInt nRanks;
    MPI_Comm_size(PETSC_COMM_WORLD, &nRanks);
    if (nRanks > 1 && !buf.empty())
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), (int)buf.size(), MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

    trapXi.assign(buf.begin(), buf.begin() + NX);
    trapR.assign(buf.begin() + NX, buf.end());
}
/**
 * Insert the combined advection and diffusion terms into the matrix.
 *
 * The coefficients of all advection terms have already been summed into
 * F1, F2 (and those of the diffusion terms into D11, D22), so the matrix
 * is filled in one pass for the advection and one for the diffusion. The
 * log lists which terms went into each pass.
 */
void AdvectionDiffusionTerm::SetMatrixElements(Matrix *mat, real_t *rhs)
{
    const bool log = TermLog::matrix();
    TermLog::clk::time_point t0;

    if (this->advectionterms.size() > 0)
    {
        if (log)
        {
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
    if (this->diffusionterms.size() > 0)
    {
        if (log)
        {
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
void AdvectionDiffusionTerm::SetVectorElements(real_t *vec, const real_t *x)
{
    const bool log = TermLog::matrix();
    TermLog::clk::time_point t0;

    if (this->advectionterms.size() > 0)
    {
        t0 = TermLog::clk::now();
        this->AdvectionTerm::SetVectorElements(vec, x);
        if (log)
            printf("[matrix]                  advection added to RHS %34s %9.3f ms\n", "", TermLog::ms(t0, TermLog::clk::now()));
    }
    if (this->diffusionterms.size() > 0)
    {
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
void AdvectionDiffusionTerm::SaveCoefficientsSFile(const string &filename)
{
    SFile *sf = SFile::Create(filename, SFILE_MODE_WRITE);
    this->SaveCoefficientsSFile(sf);
    sf->Close();
}
void AdvectionDiffusionTerm::SaveCoefficientsSFile(SFile *sf)
{
    this->AdvectionTerm::SaveCoefficientsSFile(sf);
    this->DiffusionTerm::SaveCoefficientsSFile(sf);
}
