#!/usr/bin/env python3
"""
Convergence of the toroidal kinetic runs of run_convergence_toroidal.sh.

OUTDIR holds <snapshot>/<scan>_nr<Nr>_nt<Nt>/output_*.h5 with scan = radial
(Nr varied, one time step) or time (Nr fixed, Nt steps over the same dt).
For every run it takes, at the last time, the moments

    n_re(r)    density of f_re          sum f V' dp dxi0 / V'_vol
    j_re(r)    parallel flux of f_re    sum f V' (p/gamma) xi0 dp dxi0 / V'_vol
               (proportional to the runaway current density)
    n_tail(r)  density of f_hot above p = P_TAIL (0.1 by default): the
               tail, leaving out the thermal bulk, which is not resolved at
               LEVEL <= 2 (check_physics.py --equilibrium)
    N_re       total number of runaways, sum_r dr sum f V' dp dxi0

both as the state at the last time and as the change since the first time,
and compares them with the finest run of the same scan (largest Nr, or
largest Nt). The change is what measures the solver in the radial scan:
every Nr starts from the snapshot profiles interpolated on its own grid,
so the states already differ at t = 0 (the plot shows the change). The
profiles are interpolated linearly in r onto the radial
grid of the reference, and the error is the relative L2 norm
||x - x_ref|| / ||x_ref|| (relative difference for N_re). Between two
consecutive runs of a scan the observed order is log2(e_k / e_k+1)
(meaningful when the resolution doubles and e_k+1 is well above the error
of the reference itself).

Usage:

    python3 convergence_toroidal.py OUTDIR [--plot convergence.png] [--ptail 0.1]
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
QUANTITIES = ["n_re", "j_re", "n_tail", "N_re"]


def moments(path, ptail):
    """
    Moments at the last time ('state') and their change since the first
    time ('change'). The change measures the dynamics alone: in the radial
    scan every Nr starts from the snapshot profiles interpolated on its own
    grid, so the states differ already at t = 0.
    """
    def at(h, k):
        g = h["grid"]
        r, dr, VpVol = g["r"][:], g["dr"][:], g["VpVol"][:]
        out = {"r": r}

        mg = g["runaway"]
        f = np.asarray(h["eqsys/f_re"][k], float)
        p, xi = mg["p1"][:], mg["p2"][:]
        w = mg["Vprime"][:] * mg["dp2"][:][None, :, None] * mg["dp1"][:][None, None, :]
        v = (p / np.sqrt(1 + p ** 2))[None, None, :] * xi[None, :, None]
        out["n_re"] = np.sum(f * w, axis=(1, 2)) / VpVol
        out["j_re"] = np.sum(f * w * v, axis=(1, 2)) / VpVol
        out["N_re"] = np.sum(np.sum(f * w, axis=(1, 2)) * dr)

        mg = g["hottail"]
        f = np.asarray(h["eqsys/f_hot"][k], float)
        p = mg["p1"][:]
        w = mg["Vprime"][:] * mg["dp2"][:][None, :, None] * mg["dp1"][:][None, None, :]
        out["n_tail"] = np.sum((f * w)[:, :, p > ptail], axis=(1, 2)) / VpVol
        return out

    with h5py.File(path, "r") as h:
        m0, m1 = at(h, 0), at(h, -1)
    change = {"r": m1["r"]}
    for q in QUANTITIES:
        change[q] = m1[q] - m0[q]
    return {"state": m1, "change": change}


def error(m, ref, q):
    if q == "N_re":
        return abs(m[q] - ref[q]) / abs(ref[q]) if ref[q] != 0 else np.nan
    x = np.interp(ref["r"], m["r"], m[q])
    nrm = np.linalg.norm(ref[q])
    return np.linalg.norm(x - ref[q]) / nrm if nrm > 0 else np.nan


def collect(outdir, ptail):
    """{snapshot: {scan: {(nr, nt): moments}}}"""
    data = {}
    for d in sorted(glob.glob(os.path.join(outdir, "*", "*_nr*_nt*"))):
        m = re.fullmatch(r"(radial|time)_nr(\d+)_nt(\d+)", os.path.basename(d))
        files = glob.glob(os.path.join(d, "output_*.h5"))
        if not m or not files:
            continue
        snap = os.path.basename(os.path.dirname(d))
        key = (int(m.group(2)), int(m.group(3)))
        try:
            data.setdefault(snap, {}).setdefault(m.group(1), {})[key] = moments(files[0], ptail)
        except (OSError, KeyError) as e:
            print(f"  {d}: unreadable output ({e})")
    return data


def table(snap, scan, runs, kind):
    """Print the errors of one scan ('state' or 'change'); returns (resolutions, {q: errors})."""
    idx = 0 if scan == "radial" else 1
    keys = sorted(runs, key=lambda k: k[idx])
    runs = {k: runs[k][kind] for k in keys}
    ref = runs[keys[-1]]
    label = "Nr" if scan == "radial" else "Nt"
    fixed = f"Nt = {keys[0][1]}" if scan == "radial" else f"Nr = {keys[0][0]}"
    what = "state at the last time" if kind == "state" else "change last - first time"
    print(f"\n=== {snap}  {scan} scan ({fixed}), {what}, reference {label} = {keys[-1][idx]}")
    print(f"  {label:>4}  " + "  ".join(f"{q:>10}" for q in QUANTITIES) + "     order (" + ", ".join(QUANTITIES) + ")")
    errs = {q: [error(runs[k], ref, q) for k in keys] for q in QUANTITIES}
    for i, k in enumerate(keys):
        row = "  ".join(f"{errs[q][i]:10.3e}" for q in QUANTITIES)
        order = ""
        if 0 < i < len(keys) - 1:
            o = [np.log2(errs[q][i - 1] / errs[q][i]) if errs[q][i] > 0 and errs[q][i - 1] > 0 else np.nan
                 for q in QUANTITIES]
            order = "     " + ", ".join(f"{x:4.2f}" for x in o)
        print(f"  {k[idx]:4d}  {row}{order}")
    print(f"  N_re of the reference = {ref['N_re']:.6e}")
    return [k[idx] for k in keys[:-1]], {q: e[:-1] for q, e in errs.items()}


def plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available: no plot ({path})")
        return
    snaps = sorted(results)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(len(snaps), 2, figsize=(9, 3.6 * len(snaps)), squeeze=False, facecolor=SURFACE)
    for row, snap in enumerate(snaps):
        for col, scan in enumerate(("radial", "time")):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            if scan not in results[snap]:
                ax.set_visible(False)
                continue
            xs, errs = results[snap][scan]
            for k, q in enumerate(QUANTITIES):
                ys = errs[q]
                ok = [(x, y) for x, y in zip(xs, ys) if y and y > 0 and np.isfinite(y)]
                if ok:
                    ax.plot(*zip(*ok), "o-", color=SERIES[k], linewidth=2, markersize=6, label=q)
            if len(xs) >= 2:
                e0 = max(max(v for v in errs[q] if v and np.isfinite(v)) for q in QUANTITIES
                         if any(v and np.isfinite(v) for v in errs[q]))
                ax.plot(xs, [e0 * (xs[0] / x) for x in xs], ":", color=INK2, linewidth=1, label="1st order")
                ax.plot(xs, [e0 * (xs[0] / x) ** 2 for x in xs], "--", color=INK2, linewidth=1, label="2nd order")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.set_xlabel("Nr (radial cells)" if scan == "radial" else "Nt (steps over the snapshot dt)")
            ax.set_ylabel("relative error vs finest")
            ax.set_title(f"{snap}: {scan} convergence", color=INK)
            ax.grid(True, which="major", color=GRID, linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        axes[0][-1].legend(h, l, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nPlot saved as {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("outdir", help="OUTDIR of run_convergence_toroidal.sh")
    ap.add_argument("--plot", metavar="FILE", help="save error-vs-resolution plots")
    ap.add_argument("--ptail", type=float, default=0.1, help="lower momentum of the f_hot tail moment (default 0.1)")
    args = ap.parse_args()

    data = collect(args.outdir, args.ptail)
    if not data:
        print(f"No <snapshot>/<radial|time>_nr<Nr>_nt<Nt>/output_*.h5 in {args.outdir}")
        return 1
    results = {}
    for snap in sorted(data):
        for scan in ("radial", "time"):
            runs = data[snap].get(scan, {})
            if len(runs) >= 2:
                table(snap, scan, runs, "state")
                # the plot shows the change: free of the initial-data difference
                results.setdefault(snap, {})[scan] = table(snap, scan, runs, "change")
            elif runs:
                print(f"\n=== {snap}  {scan} scan: only one run, nothing to compare")
    if args.plot and results:
        plot(args.plot, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
