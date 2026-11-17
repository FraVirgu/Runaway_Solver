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

  4. (dreami only) Times, fastest of the repeats (dreami_np<n>_r<k>):
     srun wall time, time loop, invert, rebuild, matrix, first solve (with
     the preconditioner setup) and second solve, and the speed-up and
     parallel efficiency of the time loop against the smallest n.

Snapshots named <name>_nr<Nr> (kinetic_from_snapshots.py with NR) with a
dreami run at n = Nr are also grouped by <name> into a weak-scaling table
(time and efficiency t(smallest Nr) / t, one radius per rank).

It writes every stage row to <OUTDIR>/memory_stages.csv. With --plot it
also saves a figure of peak RSS per rank and of the dreami speed-up
against n (needs matplotlib).

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


def to_seconds(text):
    """'1 minute, 2 seconds, 3 milliseconds, 4 microseconds' -> seconds."""
    units = {"hour": 3600.0, "minute": 60.0, "second": 1.0, "millisecond": 1e-3, "microsecond": 1e-6}
    t = 0.0
    for num, unit in re.findall(r"([\d.]+)\s+(hour|minute|second|millisecond|microsecond)s?\b", text):
        t += float(num) * units[unit]
    return t


def parse_timings(log, wall_file):
    """
    Timings of a complete dreami run [s] (None if missing):
      wall     srun wall time (construction + time loop + output)
      loop     'Solved equation system in ...': the time loop
      invert   'Total invert time': all the linear solves
      rebuild, matrix   from the percentages of [Timing for Solver linear]
               times the loop time (the solver's total is the loop, up
               to the time-stepper overhead)
      solve1   'iter_0:time': first solve, with the preconditioner setup
      solve2   'iter_1:time': second solve
    """
    t = dict.fromkeys(["wall", "loop", "invert", "rebuild", "matrix", "solve1", "solve2"])
    if os.path.isfile(wall_file):
        try:
            t["wall"] = float(open(wall_file).read().strip())
        except ValueError:
            pass
    pct = {}
    in_linear = False
    with open(log, errors="replace") as f:
        for line in f:
            if "Solved equation system in" in line:
                t["loop"] = to_seconds(line.split("Solved equation system in", 1)[1])
            elif line.startswith("[Timing for"):
                in_linear = line.startswith("[Timing for Solver linear]")
            elif "Total invert time:" in line:
                t["invert"] = float(line.split()[-2])
            elif line.startswith("iter_0:time"):
                t["solve1"] = float(line.split()[1])
            elif line.startswith("iter_1:time"):
                t["solve2"] = float(line.split()[1])
            elif in_linear:
                m = re.match(r"\s+(Rebuild coefficients|Construct matrix):\s+([\d.]+)%", line)
                if m:
                    pct[m.group(1)] = float(m.group(2))
    if t["loop"] is not None:
        if "Rebuild coefficients" in pct:
            t["rebuild"] = t["loop"] * pct["Rebuild coefficients"] / 100
        if "Construct matrix" in pct:
            t["matrix"] = t["loop"] * pct["Construct matrix"] / 100
    return t


def collect(outdir):
    """
    {snapshot: {run: {n: {"stages": [...], "components": {...},
                          "timings": [one dict per complete repeat]}}}}
    Memory from the first run (folder without _r<k>), timings from all.
    """
    data = {}
    for d in sorted(glob.glob(os.path.join(outdir, "*", "*_np*"))):
        m = re.fullmatch(r"(test|dreami)_np(\d+)(?:_r(\d+))?", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        run, n = m.group(1), int(m.group(2))
        first = m.group(3) is None
        snap = os.path.basename(os.path.dirname(d))
        if run == "test":
            path = os.path.join(d, f"cq_np{n}_memory.txt")
        else:
            path = os.path.join(d, "dreami.log")
        entry = data.setdefault(snap, {}).setdefault(run, {}).setdefault(
            n, {"stages": [], "components": {}, "timings": [], "path": path, "nodes": 1})
        nodes_file = os.path.join(d, "nodes.txt")
        if first and os.path.isfile(nodes_file):
            try:
                entry["nodes"] = int(open(nodes_file).read().strip())
            except ValueError:
                pass
        if not os.path.isfile(path):
            continue
        stages = parse_stages(path)
        if first:
            entry["stages"] = stages
            if run == "test":
                entry["components"] = parse_components(path)
        if run == "dreami" and stages:      # complete run
            entry["timings"].append(parse_timings(path, os.path.join(d, "wall_s.txt")))
    return data


def print_speedup(snap, runs):
    """Strong scaling of the dreami times, fastest repeat of each n."""
    best = {}
    for n, e in runs.items():
        ts = [t for t in e["timings"] if t["loop"] is not None]
        if ts:
            best[n] = (min(ts, key=lambda t: t["loop"]), len(ts))
    if not best:
        return
    ns = sorted(best)
    n0 = ns[0]
    print(f"\n  4. time [s] (fastest of the repeats); speed-up and efficiency of the time loop vs n = {n0}")
    cols = ["wall", "loop", "invert", "rebuild", "matrix", "solve1", "solve2"]
    print(f"  {'n':>5}  {'nodes':>5}  {'runs':>4}  " + "  ".join(f"{c:>9}" for c in cols) + f"  {'speed-up':>8}  {'eff.':>6}")
    for n in ns:
        t, k = best[n]
        sp = best[n0][0]["loop"] / t["loop"] if t["loop"] else float("nan")
        e = sp * n0 / n
        print(f"  {n:5d}  {runs[n]['nodes']:5d}  {k:4d}  " + "  ".join(
            f"{t[c]:9.3f}" if t[c] is not None else f"{'-':>9}" for c in cols) + f"  {sp:8.2f}  {e:6.2f}")
    print("  (loop: time stepping; solve1: first solve incl. PC setup; rebuild/matrix: from the % breakdown)")


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


def best_timing(entry):
    ts = [t for t in entry["timings"] if t["loop"] is not None]
    return (min(ts, key=lambda t: t["loop"]), len(ts)) if ts else (None, 0)


def weak_series(data):
    """
    Weak scaling: snapshots <name>_nr<Nr> grouped by <name>, each taken at
    n = Nr (one radius per rank). {name: [(nr, n, timing, n_repeats,
    peak, base)]} sorted by Nr, only groups with at least two members.
    """
    groups = {}
    for snap in data:
        m = re.fullmatch(r"(.+)_nr(\d+)", snap)
        if not m:
            continue
        nr = int(m.group(2))
        e = data[snap].get("dreami", {}).get(nr)
        if e is None:
            continue
        t, k = best_timing(e)
        if t is None:
            continue
        st = e["stages"]
        groups.setdefault(m.group(1), []).append(
            (nr, nr, t, k, st[-1]["peak_max"] if st else None, st[0]["rss_max"] if st else None))
    return {g: sorted(v, key=lambda x: x[0]) for g, v in groups.items() if len(v) > 1}


def print_weak(series):
    for name, rows in sorted(series.items()):
        t0 = rows[0][2]
        print(f"\n=== weak scaling {name}: n = Nr (one radius per rank), dreami, fastest repeat ===")
        print("  efficiency = time at the smallest Nr / time (1 = ideal: constant time)")
        cols = ["wall", "loop", "invert", "rebuild", "matrix", "solve1", "solve2"]
        print(f"  {'Nr':>5}  {'n':>5}  {'runs':>4}  " + "  ".join(f"{c:>9}" for c in cols)
              + f"  {'eff.loop':>8}  {'eff.inv':>7}  {'peak MB':>9}  {'pk-base':>8}")
        for nr, n, t, k, peak, base in rows:
            el = t0["loop"] / t["loop"] if t["loop"] else float("nan")
            ei = t0["invert"] / t["invert"] if t0["invert"] and t["invert"] else float("nan")
            pk = f"{peak:9.1f}" if peak is not None else f"{'-':>9}"
            pb = f"{peak - base:8.1f}" if peak is not None else f"{'-':>8}"
            print(f"  {nr:5d}  {n:5d}  {k:4d}  " + "  ".join(
                f"{t[c]:9.3f}" if t[c] is not None else f"{'-':>9}" for c in cols)
                + f"  {el:8.2f}  {ei:7.2f}  {pk}  {pb}")


def plot_weak(path, series):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available: no plot ({path})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for name, rows in sorted(series.items()):
        ns = [r[1] for r in rows]
        for key, style in (("loop", "o-"), ("invert", "s--")):
            if all(r[2][key] for r in rows):
                t0 = rows[0][2][key]
                axes[0].plot(ns, [r[2][key] for r in rows], style, label=f"{name}: {key}")
                axes[1].plot(ns, [t0 / r[2][key] for r in rows], style, label=f"{name}: {key}")
    axes[1].axhline(1.0, color="gray", ls=":", label="ideal")
    axes[0].set_ylabel("time [s]")
    axes[1].set_ylabel("weak efficiency")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("MPI ranks = Nr")
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nWeak-scaling plot saved as {path}")


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
    fig, axes = plt.subplots(2, len(snaps), figsize=(4.2 * len(snaps), 7.4), squeeze=False)

    # Row 2: speed-up of the dreami time loop and of the solves
    for ax, snap in zip(axes[1], snaps):
        best = {}
        for n, e in data[snap].get("dreami", {}).items():
            ts = [t for t in e["timings"] if t["loop"] is not None]
            if ts:
                best[n] = min(ts, key=lambda t: t["loop"])
        if best:
            ns = sorted(best)
            for key, style in (("loop", "o-"), ("invert", "s--"), ("wall", "^:")):
                if all(best[n][key] for n in ns):
                    ax.plot(ns, [best[ns[0]][key] / best[n][key] for n in ns], style, label=key)
            ax.plot(ns, [n / ns[0] for n in ns], ":", color="gray", label="ideal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MPI ranks")
        ax.grid(True, which="both", alpha=0.3)
    axes[1][0].set_ylabel(f"dreami speed-up")
    axes[1][-1].legend(fontsize=7)

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
                if run == "dreami":
                    print_speedup(snap, data[snap][run])

    series = weak_series(data)
    print_weak(series)

    csv_path = os.path.join(args.outdir, "memory_stages.csv")
    write_csv(csv_path, data)
    print(f"\nStage rows saved as {csv_path}")

    if args.plot:
        plot(args.plot, data)
        if series:
            root, ext = os.path.splitext(args.plot)
            plot_weak(f"{root}_weak{ext or '.png'}", series)
    return 0


if __name__ == "__main__":
    sys.exit(main())
