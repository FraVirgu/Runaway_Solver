#!/usr/bin/env python3
"""
Figures for the update slides (update_2026-11-17.tex), from the results of
run_memory_snapshots.sh (job 2859315: L2, Nr = 11, 1 time step, 1 node,
n = 1 ... 176 ranks, 2 repeats):

    fig_memory_scaling.pdf    peak RSS per rank (dreami) against n
    fig_memory_breakdown.pdf  where the memory of a rank goes (CQ_tf)
    fig_speedup.pdf           speed-up of the time loop against n

and, one per snapshot <s>,

    memory_<s>        peak memory per rank (dreami and test) against n
    speedup_<s>       speed-up of the time loop, the solve and the whole run
    time_breakdown_<s> where the time of a run goes, for each n

Usage:
    python3 make_update_plots.py [DATA] [--out DIR] [--format pdf|png]

    DATA      OUTDIR of the job (default ~/Desktop/plots/memory_L2_2859315)
    --out     where to write the figures (default: this folder)
    --format  pdf (default, for the slides) or png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from summarize_memory import collect

parser = argparse.ArgumentParser()
parser.add_argument("data", nargs="?", default=os.path.expanduser("~/Desktop/plots/memory_L2_2859315"))
parser.add_argument("--out", default=HERE)
parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
args = parser.parse_args()
DATA, OUT, EXT = args.data, os.path.expanduser(args.out), args.format
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.{EXT}"), dpi=200)
    plt.close(fig)


NR = 11

# categorical slots 1-3 of the validated default palette (fixed order)
COLORS = {"CQ_tf": "#2a78d6", "injection_t0": "#eb6834", "injection_tf": "#1baf7a"}
LABELS = {"CQ_tf": "CQ final", "injection_t0": "injection start", "injection_tf": "injection end"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

data = collect(DATA)
snaps = [s for s in COLORS if s in data]


def ranks_axis(ax):
    ax.set_xscale("log", base=2)
    ticks = [1, 11, 22, 44, 88, 176]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("MPI ranks (1 node)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.axvline(NR, color=MUTED, linewidth=1, linestyle=":")


# ---------------------------------------------------------------------
# 1. Peak memory per rank
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for s in snaps:
    byn = {n: e for n, e in data[s]["dreami"].items() if e["stages"]}
    ns = sorted(byn)
    peak = [byn[n]["stages"][-1]["peak_max"] / 1024 for n in ns]
    ax.plot(ns, peak, "-o", color=COLORS[s], linewidth=2, markersize=6, label=LABELS[s])
ref = data["CQ_tf"]["dreami"]
ns = sorted(n for n in ref if ref[n]["stages"])
p1 = ref[1]["stages"][-1]["peak_max"] / 1024
ax.plot(ns, [p1 / n for n in ns], "--", color=MUTED, linewidth=1.2, label="ideal 1/n")
ax.set_yscale("log")
ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
ax.minorticks_off()
ax.set_ylabel("peak memory per rank [GB]")
ranks_axis(ax)
ax.text(NR * 1.05, ax.get_ylim()[0] * 1.15, "n = Nr", color=MUTED, fontsize=9)
ax.legend(loc="lower left", fontsize=9)
save(fig, "fig_memory_scaling")

# ---------------------------------------------------------------------
# 2. Breakdown of the memory of the largest rank (CQ_tf)
# ---------------------------------------------------------------------
DISTRIBUTED = ("eqsys: solver (matrix allocated)", "solver: step 1 rebuild",
               "solver: step 1 matrix", "solver: step 1 solve (incl. PC setup)")
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ns_bar = [1, 11, 22, 176]
parts = {"fixed (MPI, PETSc, libraries)": [], "replicated on every rank": [],
         "distributed (matrix, solver, coefficients)": []}
for n in ns_bar:
    st = ref[n]["stages"]
    base = st[0]["rss_max"]
    dist = sum(r["added"] for r in st[1:] if r["stage"] in DISTRIBUTED)
    repl = sum(r["added"] for r in st[1:] if r["stage"] not in DISTRIBUTED)
    for k, v in zip(parts, (base, repl, dist)):
        parts[k].append(v / 1024)
colors = ["#b3b3b3", "#eb6834", "#2a78d6"]
y = range(len(ns_bar))
left = [0.0] * len(ns_bar)
for (k, vals), c in zip(parts.items(), colors):
    ax.barh(y, vals, left=left, color=c, edgecolor="white", linewidth=2, height=0.6, label=k)
    left = [l + v for l, v in zip(left, vals)]
for yi, tot in zip(y, left):
    ax.text(tot + 0.05, yi, f"{tot:.2f} GB", va="center", color=INK, fontsize=9)
ax.set_yticks(list(y))
ax.set_yticklabels([f"n = {n}" for n in ns_bar])
ax.invert_yaxis()
ax.set_xlabel("memory added per stage, largest rank [GB] (CQ final)")
ax.set_xlim(0, max(left) * 1.18)
ax.grid(True, axis="x", color=GRID, linewidth=0.6)
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig_memory_breakdown")

# ---------------------------------------------------------------------
# 3. Speed-up of the time loop
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for s in snaps:
    best = {}
    for n, e in data[s]["dreami"].items():
        ts = [t for t in e["timings"] if t["loop"] is not None]
        if ts:
            best[n] = min(t["loop"] for t in ts)
    ns = sorted(best)
    ax.plot(ns, [best[1] / best[n] for n in ns], "-o", color=COLORS[s],
            linewidth=2, markersize=6, label=LABELS[s])
ax.plot(ns, ns, "--", color=MUTED, linewidth=1.2, label="ideal")
ax.set_yscale("log", base=2)
ax.set_yticks([1, 2, 4, 8, 16, 32])
ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
ax.set_ylim(0.8, 40)
ax.set_ylabel("speed-up of the time loop")
ranks_axis(ax)
ax.legend(loc="upper left", fontsize=9)
save(fig, "fig_speedup")


# ---------------------------------------------------------------------
# 4. Per snapshot: memory, speed-up, time breakdown
# ---------------------------------------------------------------------
RUN_STYLE = {"dreami": ("#2a78d6", "o", "dreami (whole run)"),
             "test": ("#eb6834", "s", "test (construction + first rebuild)")}

for s in snaps:
    # Memory per rank: dreami and test, and the part above the fixed base
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ref_peak = None
    for run, (c, m, lab) in RUN_STYLE.items():
        byn = {n: e for n, e in data[s].get(run, {}).items() if e["stages"]}
        if not byn:
            continue
        ns = sorted(byn)
        peak = [byn[n]["stages"][-1]["peak_max"] / 1024 for n in ns]
        ax.plot(ns, peak, "-" + m, color=c, linewidth=2, markersize=6, label=lab)
        if run == "dreami":
            ref_peak = (ns, peak)
    if ref_peak:
        ns, peak = ref_peak
        ax.plot(ns, [peak[0] / n for n in ns], "--", color=MUTED, linewidth=1.2, label="ideal 1/n (dreami)")
    ax.set_yscale("log")
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_ylabel("peak memory per rank [GB]")
    ax.set_title(LABELS[s], color=INK, fontsize=11)
    ranks_axis(ax)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, f"memory_{s}")

    # Speed-up of the time loop, the solve and the whole run (fastest repeat)
    best = {}
    for n, e in data[s]["dreami"].items():
        ts = [t for t in e["timings"] if t["loop"] is not None]
        if ts:
            best[n] = min(ts, key=lambda t: t["loop"])
    ns = sorted(best)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, c, lab in (("loop", "#2a78d6", "time loop"), ("invert", "#eb6834", "linear solve"),
                        ("wall", "#1baf7a", "whole run (wall)")):
        if all(best[n][key] for n in ns):
            ax.plot(ns, [best[ns[0]][key] / best[n][key] for n in ns], "-o",
                    color=c, linewidth=2, markersize=6, label=lab)
    ax.plot(ns, ns, "--", color=MUTED, linewidth=1.2, label="ideal")
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(0.5, 40)
    ax.set_ylabel("speed-up")
    ax.set_title(LABELS[s], color=INK, fontsize=11)
    ranks_axis(ax)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, f"speedup_{s}")

    # Time breakdown of the whole run: construction (wall - loop), solve,
    # matrix, rebuild, rest of the loop
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    comps = [("construction + output", "#b3b3b3"), ("linear solve", "#2a78d6"),
             ("matrix", "#eb6834"), ("rebuild", "#1baf7a"), ("rest of loop", "#eda100")]
    vals = {k: [] for k, _ in comps}
    for n in ns:
        t = best[n]
        vals["construction + output"].append(t["wall"] - t["loop"])
        vals["linear solve"].append(t["invert"])
        vals["matrix"].append(t["matrix"])
        vals["rebuild"].append(t["rebuild"])
        vals["rest of loop"].append(max(t["loop"] - t["invert"] - t["matrix"] - t["rebuild"], 0))
    y = range(len(ns))
    left = [0.0] * len(ns)
    for k, c in comps:
        ax.barh(y, vals[k], left=left, color=c, edgecolor="white", linewidth=2, height=0.6, label=k)
        left = [l + v for l, v in zip(left, vals[k])]
    for yi, tot in zip(y, left):
        ax.text(tot + max(left) * 0.01, yi, f"{tot:.1f} s", va="center", color=INK, fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"n = {n}" for n in ns])
    ax.invert_yaxis()
    ax.set_xlabel("time [s] (fastest repeat)")
    ax.set_xlim(0, max(left) * 1.15)
    ax.set_title(LABELS[s], color=INK, fontsize=11)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, f"time_breakdown_{s}")

print(f"Figures written to {OUT} ({EXT})")
