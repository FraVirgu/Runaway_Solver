"""
Compare the collision quantities computed in serial with those computed in
parallel, after their first rebuild.

testcollisionquantity writes, in global order whatever the number of ranks,

    cq_np<ranks>_<grid>_<quantity>_<f1|f2>
    cq_np<ranks>_<grid>_<quantity>_d<unknown>_<f1|f2>

so the 'np1' files are the reference and any 'np<N>' file can be compared
with them element by element. A NaN marks a radius that no rank computed,
which is an error in a parallel run.

For each quantity the script reports

    max abs   = max |x_serial - x_parallel|
    relative  = max |x_serial - x_parallel| / max |x_serial|
    L2        = ||x_serial - x_parallel||_2 / ||x_serial||_2

and a quantity passes if its relative difference is at most the tolerance.

Usage:

    python3 compare_collision_quantities.py                 # every np<N> found
    python3 compare_collision_quantities.py --np 4
    python3 compare_collision_quantities.py --dir run --tol 1e-12
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


DREAM_ROOT = os.environ.get("DREAM_ROOT", "/home/francesco-virgulti/Desktop/DREAM")
IFACE = f"{DREAM_ROOT}/build/iface"

# Class id at the start of a PETSc binary vector file
VEC_FILE_CLASSID = 1211214


def load(filename):
    """
    Load a vector written with VecView() to a PETSc binary viewer.

    Layout (big endian): class id, n (int32), then n values (float64).
    """
    with open(filename, "rb") as f:
        classid, n = np.fromfile(f, dtype=">i4", count=2)
        if classid != VEC_FILE_CLASSID:
            raise ValueError(f"{filename}: not a PETSc binary vector")
        x = np.fromfile(f, dtype=">f8", count=n)

    if x.size != n:
        raise ValueError(f"{filename}: truncated")

    return x.astype(float)


def compare(a, b):
    """
    Returns (max abs, relative, L2) difference of b with respect to a.
    """
    d = np.abs(a - b)
    maxabs = d.max() if d.size else 0.0
    scale = np.abs(a).max() if a.size else 0.0
    norm = np.linalg.norm(a)

    relative = maxabs / scale if scale > 0 else maxabs
    l2 = np.linalg.norm(a - b) / norm if norm > 0 else np.linalg.norm(a - b)

    return maxabs, relative, l2


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--dir", default=IFACE, help="directory with the cq_np* files")
    parser.add_argument("--np", type=int, nargs="*", help="rank counts to compare with np1 (default: all found)")
    parser.add_argument("--tol", type=float, default=1e-12, help="accepted relative difference (default: 1e-12)")
    args = parser.parse_args()

    # Names of the quantities, from the serial files (.info files are
    # written by PETSc next to each vector, memory.txt by the test)
    reference = sorted(
        p for p in glob.glob(f"{args.dir}/cq_np1_*") if not p.endswith((".info", ".txt"))
    )
    if not reference:
        print(f"No serial files cq_np1_* in {args.dir}")
        return 1
    names = [os.path.basename(p)[len("cq_np1_"):] for p in reference]

    ranks = args.np
    if not ranks:
        found = set()
        for path in glob.glob(f"{args.dir}/cq_np*_*"):
            n = int(re.search(r"cq_np(\d+)_", os.path.basename(path)).group(1))
            if n != 1:
                found.add(n)
        ranks = sorted(found)
    if not ranks:
        print(f"No parallel files cq_np<N>_* (N > 1) in {args.dir}")
        return 1

    failed = 0
    width = max(len(n) for n in names)
    for n in ranks:
        print(f"\nnp1 vs np{n}  (tolerance {args.tol:g})")
        print(f"{'quantity':{width}}  {'max abs':>10}  {'relative':>10}  {'L2':>10}")
        for name in names:
            other = f"{args.dir}/cq_np{n}_{name}"
            if not os.path.exists(other):
                print(f"{name:{width}}  MISSING")
                failed += 1
                continue

            a = load(f"{args.dir}/cq_np1_{name}")
            b = load(other)
            if a.size != b.size:
                print(f"{name:{width}}  SIZE {a.size} != {b.size}")
                failed += 1
                continue

            nanA, nanB = np.isnan(a).sum(), np.isnan(b).sum()
            if nanA or nanB:
                print(f"{name:{width}}  NaN (radius not computed): {nanA} in np1, {nanB} in np{n}")
                failed += 1
                continue

            maxabs, relative, l2 = compare(a, b)
            ok = relative <= args.tol
            failed += not ok
            print(f"{name:{width}}  {maxabs:10.3e}  {relative:10.3e}  {l2:10.3e}  {'ok' if ok else 'FAIL'}")

    print(f"\n{'All quantities agree.' if failed == 0 else f'{failed} comparison(s) failed.'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
