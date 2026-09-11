
"""
Compare two DREAM runs that differ only in the linear solver.

Reference:
    output_lu.h5          Direct LU solver

Test:
    output_ilu_gmres.h5   GMRES + ILU(0)

The LU solution is used as the reference.

ERROR METRIC
------------
For each quantity, the relative error is defined as

    relative_error = max(|reference - test|) / max(|reference|)

This is a scale-relative error and is appropriate for quantities such as
distribution functions, where some cells may be extremely small.

A pointwise relative error is also calculated for diagnostic purposes:

    pointwise_error = max(
        |reference - test| / max(|reference|, |test|)
    )

The verdict is based ONLY on the scale-relative error.

PASS criterion:

    relative_error <= THRESHOLD
"""

import sys
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

IFACE = "/home/francesco-virgulti/Desktop/DREAM/build/iface"

REFERENCE_FILE = f"{IFACE}/output_lu.h5"
TEST_FILE = f"{IFACE}/output_ilu_gmres.h5"


# ---------------------------------------------------------------------------
# Quantities to compare
# ---------------------------------------------------------------------------

QUANTITIES = [
    "n_re",
    "n_hot",
    "n_cold",
    "n_tot",
    "j_ohm",
    "j_hot",
    "j_re",
    "j_tot",
    "E_field",
    "T_cold",
    "W_cold",
    "I_p",
    "psi_p",
    "psi_edge",
    "S_particle",
    "f_hot",
    "f_re",
]


# ---------------------------------------------------------------------------
# Solver / comparison settings
# ---------------------------------------------------------------------------

# This is the tolerance actually passed to PETSc.
KSP_RTOL = 1e-15

# Maximum accepted relative error.
THRESHOLD = 1e-8


# ---------------------------------------------------------------------------
# Error calculation
# ---------------------------------------------------------------------------

def calculate_error(reference, test):
    """
    Calculate errors between reference and test arrays.

    Returns:
        max_abs_error
        relative_error
        pointwise_error
        reference_scale
    """

    reference = np.asarray(reference, dtype=float)
    test = np.asarray(test, dtype=float)

    difference = np.abs(reference - test)

    # Maximum absolute difference.
    max_abs_error = np.max(difference)

    # Scale of the reference quantity.
    reference_scale = np.max(np.abs(reference))

    # Main metric used for the verdict.
    if reference_scale > 0:
        relative_error = max_abs_error / reference_scale
    else:
        relative_error = 0.0

    # Pointwise relative error, reported only as diagnostic information.
    denominator = np.maximum(np.abs(reference), np.abs(test))

    pointwise = np.zeros_like(difference)

    nonzero = denominator > 0
    pointwise[nonzero] = difference[nonzero] / denominator[nonzero]

    pointwise_error = np.max(pointwise)

    return (
        max_abs_error,
        relative_error,
        pointwise_error,
        reference_scale,
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():

    print("=" * 90)
    print("DREAM SOLVER COMPARISON")
    print("=" * 90)

    print(f"Reference : {REFERENCE_FILE}")
    print(f"Test      : {TEST_FILE}")
    print(f"Solver -ksp_rtol = {KSP_RTOL:g}")
    print(f"Relative-error tolerance = {THRESHOLD:g}")
    print()

    failures = []
    worst_error = 0.0
    worst_quantity = None

    with h5py.File(REFERENCE_FILE, "r") as reference_file, \
         h5py.File(TEST_FILE, "r") as test_file:

        print(
            f"{'quantity':<14}"
            f"{'shape':>16}"
            f"{'relative error':>18}"
            f"{'pointwise':>16}"
            f"{'max abs error':>18}"
            f"  verdict"
        )

        print("-" * 90)

        for name in QUANTITIES:

            key = f"eqsys/{name}"

            # Check that the quantity exists.
            if key not in reference_file:
                print(f"{name:<14} MISSING FROM REFERENCE")
                failures.append(name)
                continue

            if key not in test_file:
                print(f"{name:<14} MISSING FROM TEST")
                failures.append(name)
                continue

            reference = reference_file[key][:]
            test = test_file[key][:]

            # Check shape.
            if reference.shape != test.shape:
                print(
                    f"{name:<14}"
                    f"{str(reference.shape):>16}"
                    f" SHAPE MISMATCH: {test.shape}"
                )
                failures.append(name)
                continue

            # Calculate errors.
            max_abs, relative, pointwise, scale = calculate_error(
                reference,
                test
            )

            # Verdict.
            passed = relative <= THRESHOLD

            if passed:
                verdict = "PASS"
            else:
                verdict = "FAIL"
                failures.append(name)

            # Track worst quantity.
            if relative > worst_error:
                worst_error = relative
                worst_quantity = name

            print(
                f"{name:<14}"
                f"{str(reference.shape):>16}"
                f"{relative:>18.3e}"
                f"{pointwise:>16.3e}"
                f"{max_abs:>18.3e}"
                f"  {verdict}"
            )

        # -------------------------------------------------------------------
        # Distribution diagnostics
        # -------------------------------------------------------------------

        print()
        print("-" * 90)
        print("DISTRIBUTION DIAGNOSTICS")
        print("-" * 90)

        for name in ("f_hot", "f_re"):

            key = f"eqsys/{name}"

            if key not in reference_file or key not in test_file:
                continue

            reference = reference_file[key][:]
            test = test_file[key][:]

            denominator = np.maximum(
                np.abs(reference),
                np.abs(test)
            )

            pointwise = np.zeros_like(denominator)

            nonzero = denominator > 0
            pointwise[nonzero] = (
                np.abs(reference - test)[nonzero]
                / denominator[nonzero]
            )

            bad = pointwise > 1e-5

            if not bad.any():
                print(f"{name}: no cells exceed 1e-5 pointwise error")
                continue

            reference_scale = np.max(np.abs(reference))
            largest_bad_cell = np.max(np.abs(reference[bad]))

            if reference_scale > 0:
                bad_cell_ratio = largest_bad_cell / reference_scale
            else:
                bad_cell_ratio = 0.0

            print(
                f"{name}: {bad.sum()} of {reference.size} cells "
                f"have pointwise error > 1e-5"
            )

            print(
                f"  largest affected cell : {largest_bad_cell:.3e}"
            )

            print(
                f"  distribution maximum : {reference_scale:.3e}"
            )

            print(
                f"  cell/maximum ratio    : {bad_cell_ratio:.3e}"
            )

            if bad_cell_ratio < 1e-6:
                print(
                    "  -> affected cells are tiny compared with the "
                    "distribution scale; pointwise error is numerical noise."
                )

            print()

        # -------------------------------------------------------------------
        # Moment consistency check
        # -------------------------------------------------------------------

        print("-" * 90)
        print("DISTRIBUTION / MOMENT CONSISTENCY")
        print("-" * 90)

        for distribution, moment in (
            ("f_hot", "j_hot"),
            ("f_re", "j_re"),
        ):

            kd = f"eqsys/{distribution}"
            km = f"eqsys/{moment}"

            if kd not in reference_file or kd not in test_file:
                continue

            if km not in reference_file or km not in test_file:
                continue

            _, distribution_error, _, _ = calculate_error(
                reference_file[kd][:],
                test_file[kd][:]
            )

            _, moment_error, _, _ = calculate_error(
                reference_file[km][:],
                test_file[km][:]
            )

            print(
                f"{distribution:<8} relative error = "
                f"{distribution_error:.3e}"
                f"   ->   "
                f"{moment:<8} relative error = "
                f"{moment_error:.3e}"
            )

        print()
        print("=" * 90)

        # -------------------------------------------------------------------
        # Final verdict
        # -------------------------------------------------------------------

        if failures:

            print("VERDICT: FAIL")
            print()

            print(
                "The following quantities exceed the allowed "
                "relative-error tolerance:"
            )

            for name in failures:
                print(f"  - {name}")

            print()
            print(
                f"Worst relative error: "
                f"{worst_error:.3e} ({worst_quantity})"
            )

            print(
                f"Allowed relative error: "
                f"{THRESHOLD:.3e}"
            )

            print()
            print(
                "The ILU+GMRES solution does not agree with the "
                "LU reference within the specified tolerance."
            )

            return 1

        else:

            print("VERDICT: PASS")
            print()

            print(
                f"Worst relative error: "
                f"{worst_error:.3e} ({worst_quantity})"
            )

            print(
                f"Allowed relative error: "
                f"{THRESHOLD:.3e}"
            )

            print()
            print(
                "The ILU+GMRES solution agrees with the LU reference."
            )

            return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
