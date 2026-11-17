#!/usr/bin/env python3
"""
Summarize the memory scaling runs of run_memory_snapshots.sh.

The folder given (OUTDIR of the job) holds one folder per run,

    <snapshot>/test_np<n>/cq_np<n>_memory.txt   testcollisionquantity
    <snapshot>/dreami_np<n>/dreami.log           dreami -dream_log_memory

both with the table of resident set size (RSS) per stage printed by
DREAM::PrintProcessMemoryLog() (include/DREAM/Settings/ProcessMemory.hpp),
and memory.txt also with the bytes of each distributed data structure.

For every snapshot and executable it prints

  1. Scaling of the whole process, per rank:
       peak       largest peak RSS of a rank at the end of the run
       base       RSS after MPI_Init + dream_initialize (first stage): the
                  fixed part (MPI, PETSc, hypre, the executable) that no
                  distribution can remove
       peak-base  the part that should shrink as 1/n
       ideal      (peak-base at the smallest n) * n_min / n
       eff.       ideal / (peak-base): 1 = perfect, the closer to 0 the
                  more is replicated on every rank
       sum        RSS summed over the ranks at the last stage

  2. Per stage, the memory added on the worst rank ("added" column of the
     table), at the smallest and the largest n, and its scaling efficiency:
     a stage whose 'added' does not shrink with n allocates replicated data.

  3. (test only) The distributed data structures, largest rank against the
     serial size: collision values, transient weights, interpolation and
     advection/diffusion coefficients, PETSc matrix.

and writes every stage row to <OUTDIR>/memory_stages.csv. With --plot it
also saves a figure of peak RSS per rank against n (needs matplotlib).

Usage:

    python3 summarize_memory.py OUTDIR
    python3 summarize_memory.py OUTDIR --plot memory_scaling.png
"""

import argparse
import csv
import glob
import os
import re
import sys


STAGE_HEADER = "Resident set size at the end of each stage"
STAGE_END = "End of resident set size table"


def parse_stages(path):
    """
    Rows of the stage table in 'path', as a list of dicts (stage, rss_min,
    rss_max, rss_sum, added, peak_max, all in MB). Empty if the file has no
    complete table (failed run).
    """
    rows, inside, complete = [], False, False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith(STAGE_HEADER):
                rows, inside = [], True
                continue
            if not inside:
                continue
            if line.startswith(STAGE_END):
                complete = True
                inside = False
                continue
            s = line.rstrip()
            if not s or s.startswith("  ") or s.startswith("stage "):
                continue          # description and column header
            parts = s.rsplit(None, 5)
            if len(parts) != 6:
                continue
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            rows.append(dict(zip(
                ["stage", "rss_min", "rss_max", "rss_sum", "added", "peak_max"],
                [parts[0].strip()] + vals)))
    return rows if complete else []


def parse_components(path):
    """
    Per-rank data structures of memory.txt [MB]: dict name -> (max over the
    ranks, sum over the ranks, serial size or None).
    """
    names = ["collision", "transient", "interpolation", "adv/diff coeff."]
    cols = {"collision": 3, "transient": 4, "interpolation": 5, "adv/diff coeff.": 6}
    per_rank = {k: [] for k in names}
    serial = {}
    matrix = []
    section = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("Memory per rank"):
                section = "data"
                continue
            if line.startswith("PETSc matrix of the solver"):
                section = "matrix"
                continue
            if line.startswith(STAGE_HEADER):
                section = None
                continue
            parts = line.split()
            if not parts:
                continue
            if section == "data":
                if re.fullmatch(r"\d+", parts[0]) and len(parts) >= 7:
                    for k in names:
                        per_rank[k].append(float(parts[cols[k]]))
                elif parts[0] == "serial" and len(parts) >= 7:
                    for k in names:
                        serial[k] = float(parts[cols[k]])
            elif section == "matrix":
                if re.fullmatch(r"\d+", parts[0]) and len(parts) >= 4:
                    matrix.append(float(parts[3]))

    out = {}
    for k in names:
        if per_rank[k]:
            out[k] = (max(per_rank[k]), sum(per_rank[k]), serial.get(k))
    if matrix:
        out["PETSc matrix"] = (max(matrix), sum(matrix), None)
    return out


def collect(outdir):
    """
    {snapshot: {run: {n: {"stages": [...], "components": {...}}}}}
    """
    data = {}
    for d in sorted(glob.glob(os.path.join(outdir, "*", "*_np*"))):
        m = re.fullmatch(r"(test|dreami)_np(\d+)", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        run, n = m.group(1), int(m.group(2))
        snap = os.path.basename(os.path.dirname(d))
        if run == "test":
            path = os.path.join(d, f"cq_np{n}_memory.txt")
        else:
            path = os.path.join(d, "dreami.log")
        entry = {"stages": [], "components": {}, "path": path}
        if os.path.isfile(path):
            entry["stages"] = parse_stages(path)
            if run == "test":
                entry["components"] = parse_components(path)
        data.setdefault(snap, {}).setdefault(run, {})[n] = entry
    return data


def eff(ref, ref_n, val, n):
    """Strong-scaling efficiency of 'val' at n against 'ref' at ref_n."""
    if val <= 0 or ref <= 0:
        return float("nan")
    return (ref * ref_n / n) / val


def print_scaling(snap, run, runs):
    ok = {n: e for n, e in runs.items() if e["stages"]}
    failed = sorted(n for n, e in runs.items() if not e["stages"])
    print(f"\n=== {snap}  [{run}] ===")
    if failed:
        print(f"  no complete table (failed or not run) for n = {failed}")
    if not ok:
        return
    ns = sorted(ok)
    n0 = ns[0]
    base0 = ok[n0]["stages"][0]["rss_max"]
    peak0 = ok[n0]["stages"][-1]["peak_max"]
    var0 = peak0 - base0

    print(f"  1. whole process per rank [MB] (base = '{ok[n0]['stages'][0]['stage']}')")
    print(f"  {'n':>5}  {'peak':>10}  {'base':>8}  {'peak-base':>10}  {'ideal':>10}  {'eff.':>6}  {'sum':>11}")
    for n in ns:
        st = ok[n]["stages"]
        base, peak = st[0]["rss_max"], st[-1]["peak_max"]
        ideal = var0 * n0 / n
        print(f"  {n:5d}  {peak:10.1f}  {base:8.1f}  {peak - base:10.1f}  {ideal:10.1f}"
              f"  {eff(var0, n0, peak - base, n):6.2f}  {st[-1]['rss_sum']:11.1f}")

    if len(ns) > 1:
        n1 = ns[-1]
        a = {r["stage"]: r["added"] for r in ok[n0]["stages"]}
        b = {r["stage"]: r["added"] for r in ok[n1]["stages"]}
        print(f"\n  2. memory added per stage on the worst rank [MB], n = {n0} vs n = {n1}"
              f" (ideal at n = {n1}: x{n0 / n1:.3g})")
        print(f"  {'stage':<46}  {'n=' + str(n0):>10}  {'n=' + str(n1):>10}  {'eff.':>6}")
        for k, r in enumerate(ok[n0]["stages"]):
            s = r["stage"]
            if s not in b:
                continue
            if k == 0:
                print(f"  {s:<46}  {a[s]:10.1f}  {b[s]:10.1f}  {'-':>6}   (fixed part, 'base')")
                continue
            # tiny stages have no meaningful efficiency
            e = eff(a[s], n0, b[s], n1) if a[s] >= 1.0 else float("nan")
            flag = "   <- replicated?" if a[s] >= 10.0 and e < 0.5 else ""
            print(f"  {s:<46}  {a[s]:10.1f}  {b[s]:10.1f}  {e:6.2f}{flag}")

    comps = {n: ok[n]["components"] for n in ns if ok[n]["components"]}
    if comps:
        names = list(next(iter(comps.values())).keys())
        print("\n  3. distributed data, largest rank [MB] (serial = one rank before distribution)")
        print(f"  {'n':>5}  " + "  ".join(f"{k:>15}" for k in names))
        for n in ns:
            if n not in comps:
                continue
            print(f"  {n:5d}  " + "  ".join(
                f"{comps[n][k][0]:15.2f}" if k in comps[n] else f"{'-':>15}" for k in names))
        first = comps[min(comps)]
        serials = [first[k][2] for k in names]
        if any(s is not None for s in serials):
            print(f"  {'ser.':>5}  " + "  ".join(
                f"{s:15.2f}" if s is not None else f"{'-':>15}" for s in serials))


def write_csv(path, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snapshot", "run", "nprocs", "stage_index", "stage",
                    "rss_min_MB", "rss_max_MB", "rss_sum_MB", "added_MB", "peak_max_MB"])
        for snap, runs in data.items():
            for run, byn in runs.items():
                for n in sorted(byn):
                    for i, r in enumerate(byn[n]["stages"]):
                        w.writerow([snap, run, n, i, r["stage"], r["rss_min"], r["rss_max"],
                                    r["rss_sum"], r["added"], r["peak_max"]])


def plot(path, data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available: no plot ({path})")
        return

    # snapshots with at least one complete run
    snaps = sorted(s for s in data if any(e["stages"] for byn in data[s].values() for e in byn.values()))
    if not snaps:
        print("\nNo complete run: no plot")
        return
    fig, axes = plt.subplots(1, len(snaps), figsize=(4.2 * len(snaps), 3.8), squeeze=False)
    for ax, snap in zip(axes[0], snaps):
        for run, style in (("dreami", "o-"), ("test", "s--")):
            byn = {n: e for n, e in data[snap].get(run, {}).items() if e["stages"]}
            if not byn:
                continue
            ns = sorted(byn)
            peak = [byn[n]["stages"][-1]["peak_max"] for n in ns]
            var = [byn[n]["stages"][-1]["peak_max"] - byn[n]["stages"][0]["rss_max"] for n in ns]
            ax.plot(ns, peak, style, label=f"{run}: peak")
            ax.plot(ns, var, style, alpha=0.5, label=f"{run}: peak - base")
            ax.plot(ns, [var[0] * ns[0] / n for n in ns], ":", color="gray",
                    label="ideal 1/n" if run == "dreami" or "dreami" not in data[snap] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MPI ranks")
        ax.set_title(snap)
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("RSS per rank, largest rank [MB]")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved as {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("outdir", help="OUTDIR of run_memory_snapshots.sh")
    parser.add_argument("--plot", metavar="FILE", help="save a plot of peak RSS per rank against n")
    args = parser.parse_args()

    data = collect(args.outdir)
    if not data:
        print(f"No <snapshot>/<test|dreami>_np<n> folders in {args.outdir}")
        return 1

    for snap in sorted(data):
        for run in ("test", "dreami"):
            if run in data[snap]:
                print_scaling(snap, run, data[snap][run])

    csv_path = os.path.join(args.outdir, "memory_stages.csv")
    write_csv(csv_path, data)
    print(f"\nStage rows saved as {csv_path}")

    if args.plot:
        plot(args.plot, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
