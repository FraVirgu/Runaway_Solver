/**
 * Implementation of the FVM UnknownQuantity class.
 */

#include <algorithm>
#include "FVM/Equation/PrescribedParameter.hpp"
#include "FVM/UnknownQuantity.hpp"
#include "FVM/UnknownQuantityHandler.hpp"

using namespace DREAM::FVM;
using namespace std;


/**
 * Returns the part of the previous-time-step data that belongs to this
 * rank. The data itself is stored in full on every rank, so the returned
 * pointer points into that array (at the first owned element) and is
 * valid for 'nloc' elements.
 *
 * The quantity occupies the rows [offset, offset+NumberOfElements()) of
 * the equation-system matrix, and this rank owns the rows [rstart, rend).
 * The elements returned are the overlap of the two (possibly none).
 *
 * rstart: First matrix row owned by this rank.
 * rend:   One past the last matrix row owned by this rank.
 * offset: First matrix row of this quantity.
 * nloc:   On return, the number of elements owned by this rank.
 * ifirst: (optional) On return, index in the quantity of the first
 *         owned element.
 */
real_t *UnknownQuantity::GetDataPreviousLocal(
    const PetscInt rstart, const PetscInt rend, const PetscInt offset,
    len_t *nloc, len_t *ifirst
) {
    const PetscInt N  = (PetscInt)this->NumberOfElements();
    const PetscInt lo = std::max<PetscInt>(0, rstart - offset);
    const PetscInt hi = std::max<PetscInt>(lo, std::min<PetscInt>(N, rend - offset));

    *nloc = (len_t)(hi - lo);
    if (ifirst != nullptr)
        *ifirst = (len_t)lo;

    return this->GetDataPrevious() + lo;
}

/**
 * Save this unknown quantity to the given SFile.
 *
 * sf:       SFile object to use for writing the file.
 * path:     Path in the SFile to save data to.
 * saveMeta: If true, also saves grid data for the quantity.
 * current:  If true, saves only data for the most recent iteration/time step.
 */
void UnknownQuantity::SaveSFile(SFile *sf, const string& path, bool saveMeta) {
    this->data->SaveSFile(sf, this->name, path, "", saveMeta);

    string var = path + "/" + this->name;

    sf->WriteAttribute_string(var, "description", this->description);
    sf->WriteAttribute_string(var, "equation", this->description_eqn);
}

/**
 * Save this unknown quantity to the given SFile.
 *
 * sf:       SFile object to use for writing the file.
 * path:     Path in the SFile to save data to.
 * saveMeta: If true, also saves grid data for the quantity.
 * current:  If true, saves only data for the most recent iteration/time step.
 */
void UnknownQuantity::SaveSFileCurrent(SFile *sf, const string& path, bool saveMeta) {
    this->data->SaveSFileCurrent(sf, this->name, path, "", saveMeta);

    string var = path + "/" + this->name;

    sf->WriteAttribute_string(var, "description", this->description);
    sf->WriteAttribute_string(var, "equation", this->description_eqn);
}

/**
 * Set the initial value of the specified unknown quantity. If
 * the initial value has previously been specified, it is overwritten.
 *
 * val: Initial value of the quantity.
 * t0:  Initial time.
 */
void UnknownQuantity::SetInitialValue(const real_t *val, const real_t t0) {
    this->data->SetInitialValue(val, t0);
}

