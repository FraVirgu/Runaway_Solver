#!/usr/bin/env python3
#
# Plot the grid discretization and the distribution function of a
# kinetic-only run (see basic_kinetic_only.py), with the hot-tail grid
# [0, P_RE] and the runaway grid [P_RE, P_MAX] joined into a single
# (p, xi) grid.
#
# Two kinds of figure are written, all to the directory of 'figure.png':
#
#   figure.png        the grid discretization
#   f_stepNNNN.png    the distribution function at time step NNNN, for the
#                     initial condition (step 0), every STEP_EVERY-th time
#                     step and the final one
#
# Run as
#
#   $ python plot_kinetic_only.py [output.h5] [figure.png]
#
# ###################################################################

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', '~/Desktop/DREAM'))
OUT_DIR = os.path.expanduser(
    os.environ.get('KINETIC_OUT_DIR', os.path.join(DREAM_ROOT, 'build', 'iface')))

filename = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, 'output_paper.h5')
figname  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT_DIR, 'grid_single.png')
figdir   = os.path.dirname(os.path.abspath(figname))

# The distribution is plotted at the initial condition, at every
# STEP_EVERY-th time step and at the final time step
STEP_EVERY = 5

# Number of decades of f shown below its initial maximum
DECADES = 30

with h5py.File(filename, 'r') as f:
    t = f['grid/t'][:]

    # Cell centres (p1 = p, p2 = xi) and cell edges of the two grids
    p_hot,  xi_hot  = f['grid/hottail/p1'][:],   f['grid/hottail/p2'][:]
    p_re,   xi_re   = f['grid/runaway/p1'][:],   f['grid/runaway/p2'][:]
    pf_hot, xif_hot = f['grid/hottail/p1_f'][:], f['grid/hottail/p2_f'][:]
    pf_re,  xif_re  = f['grid/runaway/p1_f'][:], f['grid/runaway/p2_f'][:]

print(f'File               {filename}')
print(f'Time steps         {len(t)-1}, t = {t[0]:.4e} s to {t[-1]:.4e} s')
print(f'Hot-tail grid      Np = {p_hot.size}, Nxi = {xi_hot.size}, p in [{pf_hot[0]:.3g}, {pf_hot[-1]:.3g}]')
print(f'Runaway grid       Np = {p_re.size}, Nxi = {xi_re.size}, p in [{pf_re[0]:.3g}, {pf_re[-1]:.3g}]')

if xi_hot.size != xi_re.size or not np.allclose(xi_hot, xi_re):
    raise SystemExit('The two grids have different pitch grids and cannot be '
                     'joined into a single (p, xi) array.')

# Single grid: the runaway grid starts where the hot-tail grid ends, so the
# shared edge at p = P_RE is kept once.
p    = np.concatenate((p_hot, p_re))
p_f  = np.concatenate((pf_hot, pf_re[1:]))
xi   = xi_hot
xi_f = xif_hot

print(f'Single grid        Np = {p.size}, Nxi = {xi.size}, p in [{p_f[0]:.3g}, {p_f[-1]:.3g}]')


def load_f(f, step):
    """
    Distribution function on the single grid at the given time step, with
    shape (nxi, np). 'f' is the open output file. The distributions are
    stored as (time, radius, xi, p); there is a single radial cell.
    """
    return np.concatenate((f['eqsys/f_hot'][step, 0, :, :],
                           f['eqsys/f_re'][step, 0, :, :]), axis=1)


# =====================================================================
# Grid discretization
# =====================================================================
# Grid lines (cell edges) of the single grid, in red. Every edge is drawn in
# the lower panel, which zooms in on the interface; the upper panel shows the
# whole grid, with only every STRIDE-th momentum edge so that the lines stay
# distinguishable.
STRIDE = max(1, p_f.size // 100)
ZOOM   = (pf_hot[-1] - 20 * (pf_hot[-1] - pf_hot[-2]), pf_re[0] + 5 * (pf_re[1] - pf_re[0]))

fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

ax = axs[0]
ax.vlines(p_f[::STRIDE], xi_f[0], xi_f[-1], color='r', lw=0.4)
ax.hlines(xi_f, p_f[0], p_f[-1], color='r', lw=0.4)
ax.axvline(pf_hot[-1], color='k', lw=1, ls='--')
ax.set_xlim(p_f[0], p_f[-1])
ax.set_ylim(xi_f[0], xi_f[-1])
ax.set_xlabel(r'$p / m_e c$')
ax.set_ylabel(r'$\xi$')
ax.set_title(f'Single grid, Np = {p.size}, Nxi = {xi.size} '
             f'(every {STRIDE}th p edge drawn; dashed line: hot-tail / runaway interface)')

ax = axs[1]
ax.vlines(p_f, xi_f[0], xi_f[-1], color='r', lw=0.4)
ax.hlines(xi_f, p_f[0], p_f[-1], color='r', lw=0.4)
ax.axvline(pf_hot[-1], color='k', lw=1, ls='--')
ax.set_xlim(*ZOOM)
ax.set_ylim(xi_f[0], xi_f[-1])
ax.set_xlabel(r'$p / m_e c$')
ax.set_ylabel(r'$\xi$')
ax.set_title('Zoom on the interface, every edge drawn')

fig.savefig(figname, dpi=150)
plt.close(fig)
print(f'Grid figure        {figname}')

# =====================================================================
# Distribution function
# =====================================================================
# Time steps to plot: the initial condition, every STEP_EVERY-th time step
# and the final one. Index k of the output is the solution after k time
# steps, index 0 being the initial condition.
nSteps = len(t) - 1
steps = sorted(set(range(0, nSteps + 1, STEP_EVERY)) | {nSteps})

# Colours: one hue, light to dark, for the magnitude of f, and one colour
# per pitch for the cuts along p.
SURFACE, INK, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
CMAP = LinearSegmentedColormap.from_list('sequential_blue', [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'])
CMAP.set_bad(SURFACE)
CUT_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')

STYLE = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.titlelocation': 'left', 'legend.frameon': False,
}

# The hot-tail grid covers only p < P_RE, so the momentum axis is drawn
# logarithmically to keep both grids visible. The first edge is p = 0.
p_plot = p_f.copy()
p_plot[0] = 0.5 * p[0]
P_RE = pf_hot[-1]

# Cuts along p: parallel, perpendicular and anti-parallel to the field
cuts = ((xi.size - 1, 'parallel'), (np.argmin(np.abs(xi)), 'perpendicular'), (0, 'anti-parallel'))

with h5py.File(filename, 'r') as f, plt.rc_context(STYLE):
    # The same colour and f ranges are used for every time step, so that
    # the figures can be compared with each other.
    vmax = np.ceil(np.log10(load_f(f, 0).max()))
    vmin = vmax - DECADES

    for step in steps:
        F = load_f(f, step)

        # log10(f), with non-positive values masked out
        logF = np.ma.log10(np.ma.masked_less_equal(F, 0.0))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

        ax = axs[0]
        pc = ax.pcolormesh(p_plot, xi_f, logF, vmin=vmin, vmax=vmax, cmap=CMAP, rasterized=True)
        ax.axvline(P_RE, color=INK, lw=1)
        ax.annotate('hot-tail | runaway', (P_RE, 1), xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, color=MUTED, annotation_clip=False)
        ax.set_xscale('log')
        ax.set_xlim(p_plot[0], p_plot[-1])
        ax.set_xlabel(r'$p / m_e c$')
        ax.set_ylabel(r'$\xi$')
        label = 'initial condition' if step == 0 else 'final time step' if step == nSteps else 'time step'
        ax.set_title(f'f on the single grid, {label} {step} of {nSteps}, t = {t[step]:.3e} s', pad=18)
        fig.colorbar(pc, ax=ax, label=r'$\log_{10} f$', extend='min')

        ax = axs[1]
        for (j, name), color in zip(cuts, CUT_COLORS):
            ax.loglog(p, np.ma.masked_less_equal(F[j, :], 0.0), color=color, lw=2,
                      label=rf'$\xi = {xi[j]:.2f}$ ({name})')
        ax.axvline(P_RE, color=INK, lw=1)
        ax.set_xlim(p_plot[0], p_plot[-1])
        ax.set_ylim(10.0**vmin, 10.0**(vmax + 1))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel(r'$p / m_e c$')
        ax.set_ylabel(r'$f$')
        ax.set_title('Cuts of f along p at fixed pitch')
        ax.legend(loc='lower left')

        fig.savefig(os.path.join(figdir, f'f_step{step:04d}.png'), dpi=150)
        plt.close(fig)

print(f'Distribution       {len(steps)} figures, f_step{steps[0]:04d}.png to f_step{steps[-1]:04d}.png '
      f'(every {STEP_EVERY} time steps), in {figdir}')
