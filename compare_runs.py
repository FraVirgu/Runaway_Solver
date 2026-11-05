"""
Compare two runs of the same settings saved in two folders, step by step:
two versions of the code, or the same code on a different number of ranks.

Each folder holds what dreami writes for a run with -dream_save_system
(see OUT_DATA/<case>_np<N>_<commit>/):

    petsc_mat_np<N>_step<s>_iter<k>        matrix of the linear system
    petsc_rhs_np<N>_step<s>_iter<k>        right-hand side
    petsc_solution_np<N>_step<s>_iter<k>   solution
    petsc_solution_initial, petsc_solution_final_step
    output_*.h5                            DREAM output

PETSc writes these files in global row order whatever the number of
ranks, so the files of the two runs are paired by (kind, step, iteration),
ignoring <N>, and compared entry by entry. For each pair the script reports

    max abs   = max |a - b|
    relative  = max |a - b| / max |a|
    pattern   = (matrices) entries non-zero in only one of the two

with folder A as the reference, and then the same for the unknowns of the
DREAM output at the last time. A comparison passes if the relative
difference is at most the tolerance.

Usage:

    python3 compare_runs.py OUT_DATA/run_A OUT_DATA/run_B
    python3 compare_runs.py OUT_DATA/run_A OUT_DATA/run_B --tol 1e-10
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

from compare_matrices import load as load_matrix
from compare_solutions import load as load_vector


STEP_FILE = re.compile(r"petsc_(mat|rhs|solution)_np\d+_step(\d+)_iter(\d+)$")
KINDS = {"mat": 0, "rhs": 1, "solution": 2}


def run_files(directory):
    """
    Returns {key: path} for the PETSc files of a run, where key is
    (kind, step, iteration) for the files of a step, and the file name
    for the others (initial and final solution).
    """
    files = {}
    for path in glob.glob(f"{directory}/petsc_*"):
        name = os.path.basename(path)
        if name.endswith((".info", ".txt")):
            continue
        m = STEP_FILE.match(name)
        key = (m.group(1), int(m.group(2)), int(m.group(3))) if m else name
        files[key] = path
    return files


def sort_key(key):
    if isinstance(key, tuple):
        return (KINDS[key[0]], key[1], key[2])
    return (len(KINDS), 0, 0)


def label(key):
    if isinstance(key, tuple):
        return f"{key[0]} step {key[1]} iter {key[2]}"
    return key


def difference(a, b):
    """
    Returns (max abs, relative) difference of b with respect to a.
    """
    d = np.abs(a - b)
    maxabs = d.max() if d.size else 0.0
    scale = np.abs(a).max() if a.size else 0.0
    return maxabs, (maxabs/scale if scale > 0 else maxabs)


def compare_matrices(pa, pb):
    """
    Returns (max abs, relative, pattern) difference of matrix B with
    respect to matrix A, or raises ValueError if the shapes differ.
    """
    A, B = load_matrix(pa), load_matrix(pb)
    if A.shape != B.shape:
        raise ValueError(f"shape {A.shape} != {B.shape}")

    D = (A - B).tocsr()
    D.eliminate_zeros()
    maxabs = np.abs(D.data).max() if D.nnz else 0.0
    scale = np.abs(A.data).max() if A.nnz else 0.0

    # Entries stored in only one of the two
    SA, SB = A.copy(), B.copy()
    SA.data[:] = 1
    SB.data[:] = 1
    pattern = int(abs(SA - SB).sum())

    return maxabs, (maxabs/scale if scale > 0 else maxabs), pattern


def compare_vectors(pa, pb):
    """
    Returns (max abs, relative) difference of vector b with respect to
    vector a, or raises ValueError if the sizes differ.
    """
    a, b = load_vector(pa), load_vector(pb)
    if a.size != b.size:
        raise ValueError(f"size {a.size} != {b.size}")
    return difference(a, b)


def compare_outputs(pa, pb, tol, width):
    """
    Compare the unknowns of two DREAM output files at the last time.
    Returns the number of comparisons above the tolerance.
    """
    import h5py

    failed = 0
    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        names = sorted(set(fa["eqsys"]) & set(fb["eqsys"]))
        print(f"\nDREAM output, unknowns at the last time ({os.path.basename(pa)})")
        print(f"{'unknown':{width}}  {'max abs':>10}  {'relative':>10}")
        for q in names:
            x, y = fa["eqsys"][q], fb["eqsys"][q]
            if not hasattr(x, "shape") or not hasattr(y, "shape"):
                continue
            x, y = np.asarray(x[-1], float), np.asarray(y[-1], float)
            if x.shape != y.shape:
                print(f"{q:{width}}  SHAPE {x.shape} != {y.shape}")
                failed += 1
                continue
            maxabs, rel = difference(x, y)
            ok = rel <= tol
            failed += not ok
            print(f"{q:{width}}  {maxabs:10.3e}  {rel:10.3e}  {'ok' if ok else 'FAIL'}")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("a", help="reference run folder")
    parser.add_argument("b", help="run folder to compare")
    parser.add_argument("--tol", type=float, default=1e-12, help="accepted relative difference (default: 1e-12)")
    args = parser.parse_args()

    fa, fb = run_files(args.a), run_files(args.b)
    common = sorted(set(fa) & set(fb), key=sort_key)
    only = set(fa) ^ set(fb)
    if not common:
        print(f"No PETSc file present in both {args.a} and {args.b}")
        return 1

    print(f"A = {args.a}\nB = {args.b}\n(tolerance {args.tol:g}, relative to A)\n")
    width = max(len(label(k)) for k in common)
    print(f"{'file':{width}}  {'max abs':>10}  {'relative':>10}  {'pattern':>8}")

    failed = 0
    for key in common:
        try:
            if isinstance(key, tuple) and key[0] == "mat":
                maxabs, rel, pattern = compare_matrices(fa[key], fb[key])
                pat = f"{pattern:8d}"
            else:
                maxabs, rel = compare_vectors(fa[key], fb[key])
                pat = ""
        except ValueError as e:
            print(f"{label(key):{width}}  ERROR {e}")
            failed += 1
            continue

        ok = rel <= args.tol
        failed += not ok
        print(f"{label(key):{width}}  {maxabs:10.3e}  {rel:10.3e}  {pat:>8}  {'ok' if ok else 'FAIL'}")

    if only:
        print(f"\n{len(only)} PETSc file(s) present in only one folder (not compared)")

    outs = [sorted(glob.glob(f"{d}/output_*.h5")) for d in (args.a, args.b)]
    if outs[0] and outs[1]:
        failed += compare_outputs(outs[0][0], outs[1][0], args.tol, width)

    print(f"\n{'Identical within the tolerance.' if failed == 0 else f'{failed} comparison(s) above the tolerance.'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
