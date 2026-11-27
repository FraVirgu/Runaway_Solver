#!/usr/bin/env python3
"""
Strong scaling of one radius (Nr = 1) from a scaling_nr1_L<LEVEL>.csv
(run_scaling_nr1_cluster.sh): time to solution and speed-up vs MPI ranks for
{AMG, AMG/Euclid(1)} x {GMRES, [GMRES] + FGMRES}.

Per (snapshot, solver, n): min over the repeats of setup + solve. A run counts
only if its status is "converged" AND its true relative residual is
<= RES_MAX: the test stops on the residual of the -scale diag system, which
can differ from the true one (L4, 64 ranks: a "converged" AMG/Euclid run has
true residual 1e-2). Left panel: median over the snapshots that succeed
(band: min-max); no point where more than half of the snapshots fail.

Right panel, speed-up:
  without a reference   T(1) / T(n), of the medians of the same solver
  with --ref <file>     per snapshot T_ref / T(n), median over the snapshots;
                        T_ref = serial ILU + GMRES (one core) of that snapshot,
                        from a compare_solvers_nr1_L<LEVEL>.txt
                        (run_compare_solvers_nr1_cluster.sh)

    python3 plot_scaling_nr1.py                                       # L2
    python3 plot_scaling_nr1.py ../v5_2919957_L4/scaling_nr1_L4.csv \\
            --ref ../v5_2919957_L4/compare_solvers_nr1_L4.txt         # L4
        -> <csv without .csv>.png next to the csv, table on stdout
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='?', default=os.path.join(HERE, 'scaling_nr1_L2.csv'))
ap.add_argument('--ref', help='compare_solvers_nr1_L<LEVEL>.txt: speed-up over serial ILU + GMRES')
args = ap.parse_args()
CSV = os.path.abspath(args.csv)
RES_MAX = 1e-4

SOLVERS = [  # (key, label, color, marker, linestyle)
    (('V-cyc', 'default'), 'AMG + GMRES',                  '#2a78d6', 'o', '-'),
    (('V-cyc', 'euclid'),  'AMG/Euclid(1) + GMRES',        '#eb6834', 's', '-'),
    (('FGMRES', 'default'), '[GMRES+AMG] + FGMRES',        '#1baf7a', '^', '--'),
    (('FGMRES', 'euclid'),  '[GMRES+AMG/Euclid(1)] + FGMRES (MIRadialBJ)', '#eda100', 'D', '--'),
]


def snap(system):
    return re.sub(r'_nr\d+_L\d+$', '', system.replace('sys_toroidal_', ''))


def ok(r):
    try:
        res = float(r['res'])
    except ValueError:
        return False
    return r['status'] == 'converged' and res <= RES_MAX


rows = list(csv.DictReader(l for l in open(CSV) if not l.startswith('#')))
runs = defaultdict(list)
for r in rows:
    kind = 'V-cyc' if 'V-cyc' in r['method'] else 'FGMRES'
    runs[(kind, r['smoother'], snap(r['system']), int(r['nprocs']))].append(r)
N = sorted({int(r['nprocs']) for r in rows})
snaps = sorted({snap(r['system']) for r in rows})

# serial ILU + GMRES time per snapshot [ms]
tref = {}
if args.ref:
    cur = None
    for line in open(args.ref):
        m = re.match(r'=== (\S+)', line)
        if m:
            cur = m.group(1)
        m = re.match(r'ILU \+ GMRES .* total +(\S+) s', line)
        if m and cur:
            tref[cur] = float(m.group(1)) * 1e3

# T[key][n][snapshot] = time [ms] of the successful runs
T = {key: {n: {} for n in N} for key, *_ in SOLVERS}
its = {key: {n: [] for n in N} for key, *_ in SOLVERS}
nfail = {key: {n: 0 for n in N} for key, *_ in SOLVERS}
for key, *_ in SOLVERS:
    for n in N:
        for s in snaps:
            rs = runs[(key[0], key[1], s, n)]
            if rs and all(ok(r) for r in rs):
                T[key][n][s] = min(float(r['setup_s']) + float(r['solve_s']) for r in rs) * 1e3
                its[key][n].append(int(rs[0]['its']))
            else:
                nfail[key][n] += 1


def band(key):
    med, lo, hi = [], [], []
    for n in N:
        v = list(T[key][n].values())
        if nfail[key][n] > len(snaps) / 2:  # a "median" of a few survivors: not shown
            v = []
        med.append(np.median(v) if v else np.nan)
        lo.append(min(v) if v else np.nan)
        hi.append(max(v) if v else np.nan)
    return np.array(med), np.array(lo), np.array(hi)


def speedup(key, med):
    if not tref:
        return med[0] / med
    return np.array([np.median([tref[s] / t for s, t in T[key][n].items() if s in tref])
                     if T[key][n] and nfail[key][n] <= len(snaps) / 2 else np.nan for n in N])


# ---- table ----
print(f"{'solver':46s}" + ''.join(f'{n:>15d}' for n in N))
for key, label, *_ in SOLVERS:
    med, lo, hi = band(key)
    sp = speedup(key, med)
    print(f"{label:46s}" + ''.join(f"{m:9.0f} ms{'*' if nfail[key][n] else ' '}   " for m, n in zip(med, N)))
    print(f"{'   its (min-max), failed snapshots':46s}"
          + ''.join((f"{min(its[key][n]):>5d}-{max(its[key][n]):<4d}" if its[key][n] else '     -    ')
                    + f"{nfail[key][n]:>2d}x  " for n in N))
    print(f"{'   speed-up' + (' over serial ILU+GMRES' if tref else ' T(1)/T(n)'):46s}"
          + ''.join(f"{v:12.2f}   " for v in sp))
if tref:
    print('serial ILU + GMRES [ms]:', {s: round(t) for s, t in tref.items()})

# ---- figure ----
big = max(np.nanmax(band(k)[2]) for k, *_ in SOLVERS) > 5000
unit, div = ('s', 1e3) if big else ('ms', 1.0)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
for key, label, c, mk, ls in SOLVERS:
    med, lo, hi = band(key)
    ax1.fill_between(N, lo / div, hi / div, color=c, alpha=0.12, lw=0)
    ax1.plot(N, med / div, color=c, marker=mk, ls=ls, lw=2, ms=7, label=label)
    ax2.plot(N, speedup(key, med), color=c, marker=mk, ls=ls, lw=2, ms=7, label=label)
if tref:
    ax2.axhline(1.0, color='#8a8a85', lw=1, ls=':')
    ax2.text(N[0], 1.06, 'serial ILU(0) + GMRES', color='#6b6b66', fontsize=8)
    ax2.set_ylabel('speed-up  T(serial ILU+GMRES) / T(n)')
    ax2.set_title('Speed-up over the best serial solver (median of snapshots)', fontsize=10)
else:
    ax2.plot(N, N, color='#8a8a85', lw=1, ls=':', label='ideal')
    ax2.set_ylabel('speed-up  T(1) / T(n)')
    ax2.set_title('Speed-up of the median time', fontsize=10)
for ax in (ax1, ax2):
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel('MPI ranks (one radius split over all of them)')
    ax.grid(True, which='major', color='#e4e3df', lw=0.8)
ax1.set_ylabel(f'time to solution, setup + solve [{unit}]')
ax1.set_title(f'Time (median over the snapshots that converge, band = min-max)', fontsize=10)
ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
ax2.set_yticks(ticks)
ax2.set_yticklabels([f'{t:g}' for t in ticks])
ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
lo_sp = min(np.nanmin(speedup(k, band(k)[0])) for k, *_ in SOLVERS)
hi_sp = max(np.nanmax(speedup(k, band(k)[0])) for k, *_ in SOLVERS)
ax2.set_ylim(min(0.5, lo_sp) * 0.8, max(2.0, hi_sp if tref else N[-1]) * 1.25)
ax1.legend(fontsize=8, frameon=False, loc='best')
fig.tight_layout()
out = os.path.splitext(CSV)[0] + '.png'
fig.savefig(out, dpi=200)
print('wrote', out)
