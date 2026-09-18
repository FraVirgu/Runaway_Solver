"""
Compare the linear operator matrix assembled at the first timestep by the
serial and parallel solvers.

Reference:
    petsc_mat_serial_step1      Assembled with -n 1 (dbg_size == 1)

Test:
    petsc_mat_parallel_step1    Assembled with -n N, N > 1 (dbg_size > 1)

Both files are written by SolverLinearlyImplicit::Solve() (see
Runaway_Solver/src/Solver/SolverLinearlyImplicit.cpp) via
matrix->View(FVM::Matrix::BINARY_MATLAB, ...), guarded to fire only at
nTimeStep == 1, iter == 1. The parallel solve should assemble exactly the
same global matrix as the serial solve; any discrepancy here isolates the
bug to matrix construction (BuildMatrix / CreateSubEquation / partitioning)
rather than to the linear solve or the post-solve Store() step.

ERROR METRIC
------------
The two matrices are compared both structurally (sparsity pattern) and
numerically:

    relative_error = max(|reference - test|) / max(|reference|)

computed over the union of nonzero entries in both matrices (so an entry
that is nonzero in one matrix and missing in the other is not silently
ignored).

PASS criterion:

    shapes match
    AND sparsity patterns match (same (row, col) nonzero set)
    AND relative_error <= THRESHOLD

NOTE ON BYTE ORDER
------------------
PETSc's binary format is always written big-endian, regardless of the
platform that produced it. PetscBinaryIO.readBinaryFile() therefore returns
the matrix values (and often the index arrays) with an explicit big-endian
dtype (e.g. '>f8', '>i4'), rather than the platform's native byte order.

scipy.sparse does not accept non-native byte order dtypes: any array whose
dtype is explicitly '>' or '<' (rather than '=' / native) will raise

    ValueError: scipy.sparse does not support dtype >f8. ...

as soon as an operation that inspects .data internals is called (e.g.
.nonzero(), .tocoo()). This is fixed once, right after loading, by
converting every array to its native-byte-order equivalent before handing
it to scipy.sparse.
"""

import sys
import os
import glob

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Locate PetscBinaryIO.py (not installed as a normal importable package;
# it also imports a sibling petsc_conf.py, so its directory must be added
# to sys.path rather than loaded standalone via importlib).
# ---------------------------------------------------------------------------

def _load_petsc_binary_io():
    candidates = glob.glob("/usr/share/petsc/*/lib/petsc/bin/PetscBinaryIO.py") + \
        glob.glob(os.path.expanduser("~/miniconda3/**/lib/petsc/bin/PetscBinaryIO.py"), recursive=True)

    petsc_dir = os.environ.get("PETSC_DIR")
    if petsc_dir:
        candidates.insert(0, os.path.join(petsc_dir, "lib", "petsc", "bin", "PetscBinaryIO.py"))

    for path in candidates:
        if os.path.isfile(path):
            sys.path.insert(0, os.path.dirname(path))
            import PetscBinaryIO as module
            return module

    raise RuntimeError(
        "Could not find PetscBinaryIO.py. Set PETSC_DIR or install it "
        "on PYTHONPATH."
    )


PetscBinaryIO = _load_petsc_binary_io()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

IFACE = "/home/francesco-virgulti/Desktop/DREAM/build/iface"

REFERENCE_FILE = f"{IFACE}/petsc_mat_serial_step1"
TEST_FILE = f"{IFACE}/petsc_mat_parallel_step1"


# ---------------------------------------------------------------------------
# Comparison settings
# ---------------------------------------------------------------------------

# Maximum accepted relative error (same convention as comparison.py).
THRESHOLD = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_native(array):
    """
    Return `array` converted to its native-byte-order dtype.

    PETSc binary dumps are big-endian regardless of platform, so arrays
    coming out of PetscBinaryIO can carry an explicit '>' (or '<') byte
    order marker even on a little-endian machine. scipy.sparse rejects
    any dtype that is not native ('='), so every array that ends up
    inside a scipy sparse matrix must be normalized first.

    np.asarray(..., dtype=<native dtype>) forces an actual conversion
    (not just a reinterpreted view), which is what actually clears the
    byte-swap flag.
    """

    array = np.asarray(array)

    native_dtype = array.dtype.newbyteorder("=")

    if array.dtype != native_dtype:
        array = array.astype(native_dtype)

    return array


def load_matrix(path):
    """
    Load a PETSc binary Mat dump and return it as a scipy CSR matrix,
    with all underlying arrays normalized to native byte order.
    """

    io = PetscBinaryIO.PetscBinaryIO()

    with open(path, "rb") as fh:
        objects = io.readBinaryFile(fh, mattype="sparse")

    if not objects:
        raise IOError(f"No objects found in {path}")

    (M, N), (I, J, V) = objects[0]

    # Normalize byte order on all three CSR component arrays. Values (V)
    # are the ones that actually triggered the crash, but index arrays
    # (I, J) can carry the same big-endian marker and would fail later
    # in the same way, so they are normalized defensively too.
    I = _to_native(I)
    J = _to_native(J)
    V = _to_native(V)

    return sp.csr_matrix((V, J, I), shape=(M, N))


# ---------------------------------------------------------------------------
# Error calculation
# ---------------------------------------------------------------------------

def calculate_error(reference, test):
    """
    Calculate errors between two sparse matrices over the union of their
    nonzero patterns.

    Returns:
        max_abs_error
        relative_error
        reference_scale
        pattern_match   (bool)
        n_only_reference
        n_only_test
    """

    reference = reference.tocsr()
    test = test.tocsr()

    difference = (reference - test).tocoo()

    max_abs_error = np.max(np.abs(difference.data)) if difference.nnz > 0 else 0.0
    reference_scale = np.max(np.abs(reference.data)) if reference.nnz > 0 else 0.0

    if reference_scale > 0:
        relative_error = max_abs_error / reference_scale
    else:
        relative_error = 0.0

    ref_pattern = set(zip(*reference.nonzero()))
    test_pattern = set(zip(*test.nonzero()))

    only_reference = ref_pattern - test_pattern
    only_test = test_pattern - ref_pattern

    pattern_match = (len(only_reference) == 0) and (len(only_test) == 0)

    return (
        max_abs_error,
        relative_error,
        reference_scale,
        pattern_match,
        len(only_reference),
        len(only_test),
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():

    print("=" * 90)
    print("DREAM MATRIX COMPARISON (step 1, serial vs. parallel)")
    print("=" * 90)

    print(f"Reference : {REFERENCE_FILE}")
    print(f"Test      : {TEST_FILE}")
    print(f"Relative-error tolerance = {THRESHOLD:g}")
    print()

    for path in (REFERENCE_FILE, TEST_FILE):
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}")
            return 1

    reference = load_matrix(REFERENCE_FILE)
    test = load_matrix(TEST_FILE)

    failures = []

    # -------------------------------------------------------------------
    # Shape check
    # -------------------------------------------------------------------

    print(f"{'check':<28}{'reference':>20}{'test':>20}  verdict")
    print("-" * 90)

    shape_ok = reference.shape == test.shape
    print(
        f"{'shape':<28}"
        f"{str(reference.shape):>20}"
        f"{str(test.shape):>20}"
        f"  {'PASS' if shape_ok else 'FAIL'}"
    )
    if not shape_ok:
        failures.append("shape")
        print()
        print("VERDICT: FAIL (matrices have different dimensions, cannot compare further)")
        return 1

    nnz_ref = reference.nnz
    nnz_test = test.nnz
    nnz_ok = nnz_ref == nnz_test
    print(
        f"{'nnz':<28}"
        f"{nnz_ref:>20}"
        f"{nnz_test:>20}"
        f"  {'PASS' if nnz_ok else 'FAIL'}"
    )
    if not nnz_ok:
        failures.append("nnz")

    # -------------------------------------------------------------------
    # Value / pattern comparison
    # -------------------------------------------------------------------

    max_abs, relative, scale, pattern_match, n_only_ref, n_only_test = calculate_error(
        reference, test
    )

    print(
        f"{'sparsity pattern':<28}"
        f"{'-':>20}"
        f"{'-':>20}"
        f"  {'PASS' if pattern_match else 'FAIL'}"
    )
    if not pattern_match:
        failures.append("sparsity pattern")

    passed_values = relative <= THRESHOLD
    print(
        f"{'relative error':<28}"
        f"{relative:>20.3e}"
        f"{THRESHOLD:>20.3e}"
        f"  {'PASS' if passed_values else 'FAIL'}"
    )
    if not passed_values:
        failures.append("relative error")

    print()
    print("-" * 90)
    print("DIAGNOSTICS")
    print("-" * 90)
    print(f"  reference nnz                : {nnz_ref}")
    print(f"  test nnz                      : {nnz_test}")
    print(f"  entries only in reference     : {n_only_ref}")
    print(f"  entries only in test          : {n_only_test}")
    print(f"  max |reference - test|        : {max_abs:.3e}")
    print(f"  max |reference| (scale)       : {scale:.3e}")
    print(f"  relative error                : {relative:.3e}")

    # Show a few example mismatched entries, if any, to speed up debugging
    # of the matrix construction code.
    if n_only_ref > 0 or n_only_test > 0:
        ref_pattern = set(zip(*reference.nonzero()))
        test_pattern = set(zip(*test.nonzero()))

        only_reference = sorted(ref_pattern - test_pattern)[:10]
        only_test = sorted(test_pattern - ref_pattern)[:10]

        if only_reference:
            print()
            print("  Sample entries present only in reference (row, col):")
            for (r, c) in only_reference:
                print(f"    ({r}, {c}) = {reference[r, c]:.6e}")

        if only_test:
            print()
            print("  Sample entries present only in test (row, col):")
            for (r, c) in only_test:
                print(f"    ({r}, {c}) = {test[r, c]:.6e}")

    print()
    print("=" * 90)

    # -------------------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------------------

    if failures:
        print("VERDICT: FAIL")
        print()
        print("The following checks failed:")
        for name in failures:
            print(f"  - {name}")
        print()
        print(
            "The parallel matrix assembled at step 1 does not match the "
            "serial reference. This points to a bug in matrix construction "
            "(BuildMatrix / CreateSubEquation / row partitioning), not in "
            "the linear solve itself."
        )
        return 1
    else:
        print("VERDICT: PASS")
        print()
        print("The parallel matrix assembled at step 1 matches the serial reference.")
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())