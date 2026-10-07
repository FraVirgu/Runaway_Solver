#!/usr/bin/env python3
#
# Plot the grid discretization and the distribution function of a
# kinetic-only run on the runaway grid alone (see basic_f_re_only.py),
# where the hot-tail grid is disabled and f_re is the only distribution.
#
# Two kinds of figure are written, all to the directory of 'figure.png':
#
#   figure.png           the grid discretization
#   f_re_stepNNNN.png    the distribution function at time step NNNN, for
#                        the initial condition (step 0), every
#                        STEP_EVERY-th time step and the final one
#
# Run as
#
#   $ python plot_re_only.py [output.h5] [figure.png]
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

filename = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, 'output_f_re_only.h5')
figname  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT_DIR, 'grid_re_only.png')
figdir   = os.path.dirname(os.path.abspath(figname))

# The distribution is plotted at the initial condition, at every
# STEP_EVERY-th time step and at the final time step
STEP_EVERY = 5

# Number of decades of f shown below its initial maximum
DECADES = 30

with h5py.File(filename, 'r') as f:
    if 'hottail' in f['grid'] and 'f_hot' in f['eqsys']:
        raise SystemExit(f'{filename} has a hot-tail grid as well: use '
                         'plot_kinetic_only.py for it.')

    t = f['grid/t'][:]

    # Cell centres (p1 = p, p2 = xi) and cell edges of the runaway grid
    p,   xi   = f['grid/runaway/p1'][:],   f['grid/runaway/p2'][:]
    p_f, xi_f = f['grid/runaway/p1_f'][:], f['grid/runaway/p2_f'][:]

print(f'File               {filename}')
print(f'Time steps         {len(t)-1}, t = {t[0]:.4e} s to {t[-1]:.4e} s')
print(f'Runaway grid       Np = {p.size}, Nxi = {xi.size}, p in [{p_f[0]:.3g}, {p_f[-1]:.3g}]')

# The grid may be bi-uniform in p, with a finer spacing below P_SEP. The
# spacing changes at the edge where the cell width jumps; iSep is None for
# a uniform grid.
dp = np.diff(p_f)
jump = np.abs(np.diff(dp))
iSep = int(np.argmax(jump)) + 1 if jump.max() > 1e-6 * dp.max() else None

if iSep is None:
    print(f'  uniform in p     dp = {dp[0]:.3e}')
else:
    print(f'  bi-uniform in p  {iSep} cells with dp = {dp[0]:.3e} below p = {p_f[iSep]:.3g}, '
          f'{p.size - iSep} cells with dp = {dp[-1]:.3e} above')


def load_f(f, step):
    """
    Distribution function at the given time step, with shape (nxi, np).
    'f' is the open output file. The distribution is stored as
    (time, radius, xi, p); there is a single radial cell.
    """
    return f['eqsys/f_re'][step, 0, :, :]


# =====================================================================
# Grid discretization
# =====================================================================
# Grid lines (cell edges), in red. The upper panel shows the whole grid,
# with only every STRIDE-th momentum edge so that the lines stay
# distinguishable. Every edge is drawn in the lower panel, which zooms in
# on the change of spacing of a bi-uniform grid, or on the lower end of a
# uniform one.
STRIDE = max(1, p_f.size // 100)
if iSep is None:
    ZOOM = (p_f[0], p_f[min(25, p.size)])
else:
    ZOOM = (p_f[max(0, iSep - 20)], p_f[min(p.size, iSep + 5)])

fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

ax = axs[0]
ax.vlines(p_f[::STRIDE], xi_f[0], xi_f[-1], color='r', lw=0.4)
ax.hlines(xi_f, p_f[0], p_f[-1], color='r', lw=0.4)
if iSep is not None:
    ax.axvline(p_f[iSep], color='k', lw=1, ls='--')
ax.set_xlim(p_f[0], p_f[-1])
ax.set_ylim(xi_f[0], xi_f[-1])
ax.set_xlabel(r'$p / m_e c$')
ax.set_ylabel(r'$\xi$')
ax.set_title(f'Runaway grid, Np = {p.size}, Nxi = {xi.size} (every {STRIDE}th p edge drawn'
             + ('; dashed line: change of p spacing)' if iSep is not None else ')'))

ax = axs[1]
ax.vlines(p_f, xi_f[0], xi_f[-1], color='r', lw=0.4)
ax.hlines(xi_f, p_f[0], p_f[-1], color='r', lw=0.4)
if iSep is not None:
    ax.axvline(p_f[iSep], color='k', lw=1, ls='--')
ax.set_xlim(*ZOOM)
ax.set_ylim(xi_f[0], xi_f[-1])
ax.set_xlabel(r'$p / m_e c$')
ax.set_ylabel(r'$\xi$')
ax.set_title(('Zoom on the change of p spacing' if iSep is not None else 'Zoom on the lower end of the grid')
             + ', every edge drawn')

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

# The Maxwellian tail occupies only the lowest part of the momentum range,
# so the momentum axis is drawn logarithmically to keep it visible. This
# needs p_min > 0, which holds for a runaway grid that starts at P_MIN.
logP = p_f[0] > 0


def set_p_axis(ax):
    """
    Set the limits, scale and label of the momentum axis. A logarithmic
    axis only has labelled ticks at the powers of ten, so both ends of the
    grid, p_min and p_max, are labelled explicitly.
    """
    if logP:
        decades = 10.0**np.arange(np.ceil(np.log10(p_f[0])), np.floor(np.log10(p_f[-1])) + 1)
        ticks = [p_f[0]] + [d for d in decades if 1.5 * p_f[0] < d < p_f[-1] / 1.5] + [p_f[-1]]
        ax.set_xscale('log')
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{v:.3g}' for v in ticks])
    ax.set_xlim(p_f[0], p_f[-1])
    ax.set_xlabel(r'$p / m_e c$')


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
        pc = ax.pcolormesh(p_f, xi_f, logF, vmin=vmin, vmax=vmax, cmap=CMAP, rasterized=True)
        set_p_axis(ax)
        ax.set_ylabel(r'$\xi$')
        label ='initial condition' if step == 0 else 'final time step' if step == nSteps else 'time step'
        ax.set_title(f'f_re, {label} {step} of {nSteps}, t = {t[step]:.3e} s')
        fig.colorbar(pc, ax=ax, label=r'$\log_{10} f$', extend='min')

        ax = axs[1]
        for (j, name), color in zip(cuts, CUT_COLORS):
            ax.plot(p, np.ma.masked_less_equal(F[j, :], 0.0), color=color, lw=2,
                    label=rf'$\xi = {xi[j]:.2f}$ ({name})')
        ax.set_yscale('log')
        set_p_axis(ax)
        ax.set_ylim(10.0**vmin, 10.0**(vmax + 1))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylabel(r'$f$')
        ax.set_title('Cuts of f_re along p at fixed pitch')
        ax.legend(loc='upper right')

        fig.savefig(os.path.join(figdir, f'f_re_step{step:04d}.png'), dpi=150)
        plt.close(fig)

print(f'Distribution       {len(steps)} figures, f_re_step{steps[0]:04d}.png to f_re_step{steps[-1]:04d}.png '
      f'(every {STEP_EVERY} time steps), in {figdir}')
