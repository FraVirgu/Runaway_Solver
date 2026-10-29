#!/usr/bin/env python3
#
# Plot the distribution functions of a radially resolved kinetic run at a
# single time step, as a function of radius (see
# examples/kinetic_from_snapshots.py, which runs one time step).
#
# The hot-tail and runaway grids are plotted separately, since their pitch
# grids differ. Three figures are written, all to the directory of
# 'prefix', with names starting with 'prefix':
#
#   <prefix>_hot.png    log10 f_hot(p, xi), one panel per radius
#   <prefix>_re.png     log10 f_re(p, xi),  one panel per radius
#   <prefix>_cuts.png   f along p at the pitch closest to xi = +1, one line
#                       per radius, and its relative change over the step
#
# Run as
#
#   $ python plot_kinetic_radial_step.py [output.h5] [prefix] [step]
#
# where step is the time index to plot (default: the last one, -1).
#
# ###################################################################

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'DREAM')))
OUT_DIR = os.path.expanduser(
    os.environ.get('KINETIC_OUT_DIR', os.path.join(DREAM_ROOT, 'build', 'iface')))

filename = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, 'output_kinetic_CQ_tf_L2.h5')
prefix   = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(filename)[0]
step     = int(sys.argv[3]) if len(sys.argv) > 3 else -1

# Number of decades of f shown below its maximum
DECADES = 30

with h5py.File(filename, 'r') as f:
    t = f['grid/t'][:]
    r = f['grid/r'][:]
    step = step % t.size

    grids = {}
    for name, key in (('hot', 'hottail'), ('re', 'runaway')):
        g = f['grid'][key]
        grids[name] = {
            'p': g['p1'][:], 'xi': g['p2'][:], 'p_f': g['p1_f'][:], 'xi_f': g['p2_f'][:],
            # stored as (time, radius, xi, p)
            'f':  f['eqsys'][f'f_{name}'][step, :, :, :],
            'f0': f['eqsys'][f'f_{name}'][0, :, :, :],
        }

print(f'File               {filename}')
print(f'Time step          {step} of {t.size-1}, t = {t[step]:.4e} s (dt = {t[step]-t[0]:.4e} s from the initial condition)')
print(f'Radial grid        Nr = {r.size}, r in [{r[0]:.3g}, {r[-1]:.3g}] m')
for name, g in grids.items():
    print(f'{name:3s} grid           Np = {g["p"].size}, Nxi = {g["xi"].size}, p in [{g["p_f"][0]:.3g}, {g["p_f"][-1]:.3g}]')

# Colours: one hue, light to dark, for the magnitude of f, and the same hue
# family, light (core) to dark (edge), for the radius.
SURFACE, INK, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
CMAP = LinearSegmentedColormap.from_list('sequential_blue', [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'])
CMAP.set_bad(SURFACE)

# The lightest steps are too faint for lines on the light surface
R_CMAP = LinearSegmentedColormap.from_list('radius', [CMAP(x) for x in np.linspace(0.3, 1.0, 64)])
R_NORM = Normalize(r[0], r[-1])

STYLE = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.titlelocation': 'left', 'legend.frameon': False,
}


def masked_log(F):
    """log10(F), with non-positive values masked out."""
    return np.ma.log10(np.ma.masked_less_equal(F, 0.0))


# =====================================================================
# f(p, xi), one panel per radius
# =====================================================================
ncol = 4
nrow = int(np.ceil(r.size / ncol))

with plt.rc_context(STYLE):
    for name, label, tex in (('hot', 'f_hot', r'f_{\rm hot}'), ('re', 'f_re', r'f_{\rm re}')):
        g = grids[name]

        # The same colour range at every radius, so that panels compare
        vmax = np.ceil(np.log10(g['f'].max())) if g['f'].max() > 0 else 0.0
        vmin = vmax - DECADES

        # The hot-tail grid starts at p = 0, so its first edge is moved for
        # the logarithmic momentum axis
        p_plot = g['p_f'].copy()
        if p_plot[0] <= 0:
            p_plot[0] = 0.5 * g['p'][0]

        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                                sharex=True, sharey=True, constrained_layout=True)
        for ir, ax in enumerate(axs.flat):
            if ir >= r.size:
                ax.set_visible(False)
                continue
            pc = ax.pcolormesh(p_plot, g['xi_f'], masked_log(g['f'][ir]), vmin=vmin, vmax=vmax,
                               cmap=CMAP, rasterized=True)
            ax.set_xscale('log')
            ax.set_xlim(p_plot[0], p_plot[-1])
            ax.set_title(f'r = {r[ir]:.3f} m', fontsize=10)
            if ir % ncol == 0:
                ax.set_ylabel(r'$\xi$')
            if ir >= r.size - ncol:
                # Bottom panel of its column (the last row may be incomplete)
                ax.set_xlabel(r'$p / m_e c$')
                ax.tick_params(labelbottom=True)

        fig.colorbar(pc, ax=axs, label=rf'$\log_{{10}}\, {tex}$', extend='min', shrink=0.8)
        fig.suptitle(f'{label} at time step {step}, t = {t[step]:.3e} s', x=0.01, ha='left')

        out = f'{prefix}_{name}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'Figure             {out}')

# =====================================================================
# Cuts along p at xi ~ +1, one line per radius
# =====================================================================
# The two grids are joined along p only: each contributes its own pitch
# closest to xi = +1, so the cut is continuous at p = P_RE.
def cut(key):
    parts = []
    for g in grids.values():
        j = np.argmax(g['xi'])
        parts.append(g[key][:, j, :])
    return np.concatenate(parts, axis=1)

p_cut  = np.concatenate([g['p'] for g in grids.values()])
F_cut  = cut('f')
F0_cut = cut('f0')
P_RE   = grids['hot']['p_f'][-1]
xi_par = (grids['hot']['xi'].max(), grids['re']['xi'].max())

with np.errstate(divide='ignore', invalid='ignore'):
    change = np.abs(F_cut - F0_cut) / np.abs(F0_cut)

with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    for ir in range(r.size):
        color = R_CMAP(R_NORM(r[ir]))
        axs[0].loglog(p_cut, np.ma.masked_less_equal(F_cut[ir], 0.0), color=color, lw=2)
        axs[1].loglog(p_cut, np.ma.masked_less_equal(change[ir], 0.0), color=color, lw=2)

    for ax in axs:
        ax.axvline(P_RE, color=INK, lw=1)
        ax.grid(True)
        ax.set_axisbelow(True)
    axs[0].annotate('hot-tail | runaway', (P_RE, 1), xycoords=('data', 'axes fraction'),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, color=MUTED, annotation_clip=False)

    vmax = np.ceil(np.log10(F_cut.max())) if F_cut.max() > 0 else 0.0
    axs[0].set_ylim(10.0**(vmax - DECADES), 10.0**(vmax + 1))
    axs[0].set_ylabel(r'$f$')
    axs[0].set_title(f'f along p at the most parallel pitch (xi = {xi_par[0]:.3f} hot, {xi_par[1]:.4f} runaway), '
                     f'time step {step}, t = {t[step]:.3e} s', pad=18)

    axs[1].set_xlabel(r'$p / m_e c$')
    axs[1].set_ylabel(r'$|f - f_0| \,/\, f_0$')
    axs[1].set_title('Relative change from the initial condition')

    fig.colorbar(ScalarMappable(norm=R_NORM, cmap=R_CMAP), ax=axs, label='r (m)', shrink=0.8)

    out = f'{prefix}_cuts.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Figure             {out}')
