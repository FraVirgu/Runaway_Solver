"""
Compare the equation-system matrix built in serial with the one built in
parallel.

SolverLinearlyImplicit writes the matrix of every iteration to

    petsc_mat_np<ranks>_step<N>_iter<k>

PETSc stores these files in global row order whatever the number of ranks,
so the 'np1' file is the reference and any 'np<N>' file can be compared
with it entry by entry.

For each of the first steps the script reports

    max abs   = max |A_serial - A_parallel|
    relative  = max |A_serial - A_parallel| / max |A_serial|
    pattern   = number of entries that are non-zero in only one of the two

and the entries with the largest difference. A step passes if the relative
difference is at most the tolerance.

Usage:

    python3 compare_matrices.py                 # every np<N> found, steps 1..5
    python3 compare_matrices.py --np 4          # only the 4-rank run
    python3 compare_matrices.py --steps 2 --tol 1e-10
    python3 compare_matrices.py --np1 100       # also give (line, i) of a row
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import scipy.sparse as sparse


DREAM_ROOT = os.environ.get("DREAM_ROOT", "/home/francesco-virgulti/Desktop/DREAM")
IFACE = f"{DREAM_ROOT}/build/iface"

# Class id at the start of a PETSc binary matrix file
MAT_FILE_CLASSID = 1211216


def load(filename):
    """
    Load a matrix written with MatView() to a PETSc binary viewer.

    Layout (big endian): class id, M, N, nnz (int32), then the number of
    entries of each row (M int32), the column of each entry (nnz int32)
    and the value of each entry (nnz float64).
    """
    with open(filename, "rb") as f:
        classid, M, N, nnz = np.fromfile(f, dtype=">i4", count=4)
        if classid != MAT_FILE_CLASSID:
            raise ValueError(f"{filename}: not a PETSc binary matrix")

        rowlen = np.fromfile(f, dtype=">i4", count=M)
        cols = np.fromfile(f, dtype=">i4", count=nnz)
        vals = np.fromfile(f, dtype=">f8", count=nnz)

    if rowlen.sum() != nnz or vals.size != nnz:
        raise ValueError(f"{filename}: truncated or written with 64-bit indices")

    indptr = np.concatenate(([0], np.cumsum(rowlen, dtype=np.int64)))
    A = sparse.csr_matrix(
        (vals.astype(float), cols.astype(np.int32), indptr), shape=(M, N))
    A.sum_duplicates()

    return A


def load_blocks(directory):
    """
    Returns [(name, first row, one past last row), ...] for the kinetic
    blocks of the matrix, from petsc_block_sizes.txt (empty if missing).
    """
    path = f"{directory}/petsc_block_sizes.txt"
    if not os.path.exists(path):
        return []

    sizes = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                sizes[parts[0]] = int(parts[1])

    blocks, start = [], 0
    for key, name in (("Nhot", "f_hot"), ("Nre", "f_re")):
        if sizes.get(key, 0) > 0:
            blocks.append((name, start, start + sizes[key]))
            start += sizes[key]

    return blocks


def describe(index, blocks, np1):
    """
    Name the unknown a matrix row (or column) belongs to.
    """
    for name, lo, hi in blocks:
        if lo <= index < hi:
            k = index - lo
            if np1:
                return f"{name}[{k}] (line {k // np1}, i {k % np1})"
            return f"{name}[{k}]"

    return "fluid" if blocks else ""


def compare(A, B, blocks, np1, nworst):
    """
    Compare two matrices. Returns the relative difference.
    """
    # Entries stored as explicit zeros are not part of the pattern
    A.eliminate_zeros()
    B.eliminate_zeros()

    pattern = (A != 0) != (B != 0)
    D = abs(A - B).tocoo()

    scale = abs(A).max()
    maxabs = D.data.max() if D.nnz > 0 else 0.0
    relative = maxabs / scale if scale > 0 else maxabs

    print(f"    non-zeros   : serial {A.nnz}, parallel {B.nnz}")
    print(f"    pattern     : {pattern.nnz} entries non-zero in only one matrix")
    print(f"    max |A|     : {scale:.6e}")
    print(f"    max abs     : {maxabs:.6e}")
    print(f"    relative    : {relative:.6e}")

    if maxabs > 0:
        order = np.argsort(D.data)[::-1][:nworst]
        print(f"    largest differences:")
        for k in order:
            r, c = int(D.row[k]), int(D.col[k])
            print(f"      ({r}, {c})  serial {A[r, c]: .6e}  parallel {B[r, c]: .6e}"
                  f"  diff {D.data[k]:.3e}  row {describe(r, blocks, np1)}")

    return relative


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--dir", default=IFACE, help="directory with the petsc_mat_* files")
    parser.add_argument("--np", type=int, nargs="*", help="rank counts to compare with np1 (default: all found)")
    parser.add_argument("--steps", type=int, default=5, help="compare the steps 1..STEPS (default: 5)")
    parser.add_argument("--iter", type=int, default=1, help="iteration of each step (default: 1)")
    parser.add_argument("--tol", type=float, default=1e-12, help="accepted relative difference (default: 1e-12)")
    parser.add_argument("--np1", type=int, default=0, help="momentum points per line, to locate a row on the grid")
    parser.add_argument("--worst", type=int, default=5, help="number of largest differences to list (default: 5)")
    args = parser.parse_args()

    name = lambda n, step: f"{args.dir}/petsc_mat_np{n}_step{step}_iter{args.iter}"

    ranks = args.np
    if not ranks:
        found = set()
        for path in glob.glob(f"{args.dir}/petsc_mat_np*_step*_iter{args.iter}"):
            found.add(int(re.search(r"petsc_mat_np(\d+)_", path).group(1)))
        ranks = sorted(found - {1})

    print("=" * 90)
    print("SERIAL / PARALLEL MATRIX COMPARISON")
    print("=" * 90)
    print(f"Directory : {args.dir}")
    print(f"Reference : np1")
    print(f"Compared  : {', '.join(f'np{n}' for n in ranks) if ranks else '(none found)'}")
    print(f"Steps     : 1..{args.steps} (iteration {args.iter})")
    print(f"Tolerance : {args.tol:g}")

    if not ranks:
        return 1

    blocks = load_blocks(args.dir)
    failures = []

    for step in range(1, args.steps + 1):
        print()
        print("-" * 90)
        print(f"STEP {step}")
        print("-" * 90)

        if not os.path.exists(name(1, step)):
            print(f"  MISSING {os.path.basename(name(1, step))}")
            failures.append(f"step {step}: no serial matrix")
            continue

        A = load(name(1, step))
        print(f"  np1: {A.shape[0]} x {A.shape[1]}")

        for n in ranks:
            label = f"step {step}, np{n}"
            print(f"  np{n}:")

            if not os.path.exists(name(n, step)):
                print(f"    MISSING {os.path.basename(name(n, step))}")
                failures.append(f"{label}: no matrix")
                continue

            B = load(name(n, step))
            if B.shape != A.shape:
                print(f"    SHAPE MISMATCH: {B.shape[0]} x {B.shape[1]}")
                failures.append(f"{label}: shape mismatch")
                continue

            relative = compare(A, B, blocks, args.np1, args.worst)
            if relative <= args.tol:
                print("    verdict     : PASS")
            else:
                print("    verdict     : FAIL")
                failures.append(f"{label}: relative difference {relative:.3e}")

    print()
    print("=" * 90)
    if failures:
        print("VERDICT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
