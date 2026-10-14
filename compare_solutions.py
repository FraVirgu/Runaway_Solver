"""
Compare the solution computed in serial with the one computed in parallel,
time step by time step.

SolverLinearlyImplicit writes the solution of every time step to

    petsc_solution_np<ranks>_step<N>_iter<k>

(f_hot first, then f_re, in global order whatever the number of ranks), so
the 'np1' file is the reference and any 'np<N>' file can be compared with
it element by element.

For each step and for each of f_hot and f_re the script reports

    max abs   = max |x_serial - x_parallel|
    relative  = max |x_serial - x_parallel| / max |x_serial|
    L2        = ||x_serial - x_parallel||_2 / ||x_serial||_2

so that the growth of the difference over the steps can be followed.
A step passes if the relative difference of both blocks is at most the
tolerance.

Usage:

    python3 compare_solutions.py                    # every np<N> found, steps 1..10
    python3 compare_solutions.py --np 3
    python3 compare_solutions.py --steps 5 --tol 1e-8
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


def load_blocks(directory, n):
    """
    Returns [(name, first, one past last), ...] for the blocks of the
    solution vector, from petsc_block_sizes.txt (one block if missing).
    """
    path = f"{directory}/petsc_block_sizes.txt"
    sizes = {}
    if os.path.exists(path):
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

    if start != n:
        return [("x", 0, n)]

    return blocks


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
    parser.add_argument("--dir", default=IFACE, help="directory with the petsc_solution_* files")
    parser.add_argument("--np", type=int, nargs="*", help="rank counts to compare with np1 (default: all found)")
    parser.add_argument("--steps", type=int, default=10, help="compare the steps 1..STEPS (default: 10)")
    parser.add_argument("--iter", type=int, default=1, help="iteration of each step (default: 1)")
    parser.add_argument("--tol", type=float, default=1e-8, help="accepted relative difference (default: 1e-8)")
    args = parser.parse_args()

    name = lambda n, step: f"{args.dir}/petsc_solution_np{n}_step{step}_iter{args.iter}"

    ranks = args.np
    if not ranks:
        found = set()
        for path in glob.glob(f"{args.dir}/petsc_solution_np*_step*_iter{args.iter}"):
            found.add(int(re.search(r"petsc_solution_np(\d+)_", path).group(1)))
        ranks = sorted(found - {1})

    print("=" * 90)
    print("SERIAL / PARALLEL SOLUTION COMPARISON")
    print("=" * 90)
    print(f"Directory : {args.dir}")
    print(f"Reference : np1")
    print(f"Compared  : {', '.join(f'np{n}' for n in ranks) if ranks else '(none found)'}")
    print(f"Steps     : 1..{args.steps} (iteration {args.iter})")
    print(f"Tolerance : {args.tol:g} (relative)")

    if not ranks:
        return 1

    failures = []
    for n in ranks:
        print()
        print("-" * 90)
        print(f"np{n} against np1")
        print("-" * 90)
        print(f"{'step':>4}  {'block':<6}{'max |x|':>14}{'max abs':>14}{'relative':>14}{'L2':>14}  verdict")

        for step in range(1, args.steps + 1):
            if not os.path.exists(name(1, step)) or not os.path.exists(name(n, step)):
                missing = [os.path.basename(name(k, step)) for k in (1, n) if not os.path.exists(name(k, step))]
                print(f"{step:>4}  MISSING {', '.join(missing)}")
                failures.append(f"np{n}, step {step}: missing file")
                continue

            a, b = load(name(1, step)), load(name(n, step))
            if a.size != b.size:
                print(f"{step:>4}  SIZE MISMATCH: {a.size} and {b.size}")
                failures.append(f"np{n}, step {step}: size mismatch")
                continue

            for block, lo, hi in load_blocks(args.dir, a.size):
                maxabs, relative, l2 = compare(a[lo:hi], b[lo:hi])
                passed = relative <= args.tol
                print(f"{step:>4}  {block:<6}{np.abs(a[lo:hi]).max():>14.6e}{maxabs:>14.3e}"
                      f"{relative:>14.3e}{l2:>14.3e}  {'PASS' if passed else 'FAIL'}")
                if not passed:
                    failures.append(f"np{n}, step {step}, {block}: relative difference {relative:.3e}")

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
