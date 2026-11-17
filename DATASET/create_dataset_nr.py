#!/usr/bin/env python3
#
# Create the dreami settings to regenerate the dataset (the two simulations
# output_injection_start_M4.h5 and output_CQ_M4.h5, made by
# run_single_pellet.py, case S3) on finer radial grids.
#
# The original runs use nr = 11 radii. Here nr grows exponentially:
#
#   nr_k = NR0 * FACTOR**k,   k = 0, ..., K        (default 11, 22, 44, 88, 176)
#
# Everything else is taken unchanged from the settings stored inside the two
# original outputs (equilibrium, SPI pellet, ions, transport, time steps,
# solver), so the new runs differ from the originals only in nr. The only
# setting tied to the 11 radii is the initial ohmic current profile j_ohm,
# given on the 11 original cell centres: it is interpolated linearly (held
# constant beyond the first/last centre) onto the new cell centres; DREAM then
# rescales it to the total current Ip0 as before. The SPI shift in grid cells
# (nbrShiftGridCell) is scaled by nr/11 (it is 0 in the original runs).
#
# For every nr the folder nr_<nr>/ gets
#
#   settings_injection.h5   pellet injection, from t = 0
#                           -> output_injection_start_nr<nr>.h5
#   settings_CQ.h5          current quench, initialized from the last time
#                           step of output_injection_start_nr<nr>.h5
#                           -> output_CQ_nr<nr>.h5
#   run.sh                  runs the two in order
#
# The numerical ITER equilibrium ITER_H26_LUKE.h5 is not part of the release:
# it is looked for in DATASET/ (or $EQUILIBRIUM); copy it there before running.
#
# Usage:
#   $ python create_dataset_nr.py                  # nr = 11, 22, 44, 88, 176
#   $ K=6 python create_dataset_nr.py              # ... up to 11 * 2**6 = 704
#   $ NR0=11 FACTOR=3 K=3 python create_dataset_nr.py    # 11, 33, 99, 297
#   $ NR_LIST="22 44" python create_dataset_nr.py  # explicit list
#   $ cd nr_44 && ./run.sh                         # with dreami on the PATH
#
# Then extract the snapshots of the new outputs as extract_snapshots.py does
# for the original ones.

import os
import sys

import h5py
import numpy as np

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', os.path.join(DATASET_DIR, '..', '..', 'DREAM')))
sys.path.append(os.path.join(DREAM_ROOT, 'py'))

from DREAM.DREAMSettings import DREAMSettings
import DREAM.DREAMIO as DREAMIO

INJECTION = 'output_injection_start_M4.h5'
CQ = 'output_CQ_M4.h5'
EQUILIBRIUM = os.path.abspath(os.path.expanduser(
    os.environ.get('EQUILIBRIUM', os.path.join(DATASET_DIR, 'ITER_H26_LUKE.h5'))))
OUT_ROOT = os.path.abspath(os.path.expanduser(os.environ.get('OUT_ROOT', DATASET_DIR)))

NR_ORIG = 11

# Exponential sequence of radial resolutions
NR0    = int(os.environ.get('NR0', NR_ORIG))
FACTOR = int(os.environ.get('FACTOR', 2))
K      = int(os.environ.get('K', 4))
if os.environ.get('NR_LIST'):
    NR_LIST = [int(n) for n in os.environ['NR_LIST'].split()]
else:
    NR_LIST = [NR0 * FACTOR**k for k in range(K + 1)]


def cell_centres(a, nr):
    """Cell centres of the uniform grid r_f = linspace(0, a, nr+1) used by DREAM."""
    r_f = np.linspace(0, a, nr + 1)
    return 0.5 * (r_f[1:] + r_f[:-1])


def refine(ds, nr, output, folder):
    """Set nr (and everything that depends on it) in the settings ds."""
    a = float(np.ravel(ds.radialgrid.a)[0])

    ds.radialgrid.setNr(nr)
    ds.radialgrid.num_filename = EQUILIBRIUM

    # Initial ohmic current: original 11 cell centres -> new cell centres
    j = ds.eqsys.j_ohm
    if j.jpres0 is not None and np.size(j.jpres0_radius) > 1:
        r_new = cell_centres(a, nr)
        j.jpres0 = np.interp(r_new, np.ravel(j.jpres0_radius), np.ravel(j.jpres0))
        j.jpres0_radius = r_new

    # SPI shift counted in grid cells: keep the same shift in metres
    spi = ds.eqsys.spi
    if getattr(spi, 'nbrShiftGridCell', None) is not None:
        spi.nbrShiftGridCell = np.rint(np.asarray(spi.nbrShiftGridCell) * nr / NR_ORIG).astype(np.int64)

    ds.output.setFilename(output)
    return ds


def save(ds, filename):
    if os.path.isfile(EQUILIBRIUM):
        ds.save(filename)
    else:
        # Verification fails while the equilibrium file is missing
        DREAMIO.SaveDictAsHDF5(filename, ds.todict(verify=False))


def main():
    for fname in (INJECTION, CQ):
        if not os.path.isfile(os.path.join(DATASET_DIR, fname)):
            sys.exit(f'{fname} not found in {DATASET_DIR}; run download_datasets.py first.')

    with h5py.File(os.path.join(DATASET_DIR, INJECTION), 'r') as f:
        t_inj = f['grid/t'][:]
    with h5py.File(os.path.join(DATASET_DIR, CQ), 'r') as f:
        t_cq = f['grid/t'][:]

    print(f'nr = {NR_LIST}')
    for nr in NR_LIST:
        folder = os.path.join(OUT_ROOT, f'nr_{nr}')
        os.makedirs(folder, exist_ok=True)
        out_inj = f'output_injection_start_nr{nr}.h5'
        out_cq  = f'output_CQ_nr{nr}.h5'

        # Injection: from t = 0, as the original run
        ds = DREAMSettings(os.path.join(DATASET_DIR, INJECTION), path='settings')
        refine(ds, nr, out_inj, folder)
        save(ds, os.path.join(folder, 'settings_injection.h5'))

        # Current quench: from the end of the injection run on the same grid
        ds = DREAMSettings(os.path.join(DATASET_DIR, CQ), path='settings')
        refine(ds, nr, out_cq, folder)
        ds.fromOutput(out_inj, relpath=True, timeindex=-1)
        save(ds, os.path.join(folder, 'settings_CQ.h5'))

        run = os.path.join(folder, 'run.sh')
        with open(run, 'w') as fh:
            fh.write('#!/bin/bash\n'
                     f'# Dataset with nr = {nr}: injection, then current quench\n'
                     'set -e\n'
                     'cd "$(dirname "${BASH_SOURCE[0]}")"\n'
                     'DREAMI=${DREAMI:-dreami}\n'
                     '$DREAMI settings_injection.h5\n'
                     '$DREAMI settings_CQ.h5\n')
        os.chmod(run, 0o755)

        print(f'nr = {nr:5d}  dr = {1e3 * float(np.ravel(ds.radialgrid.a)[0]) / nr:7.2f} mm  -> {folder}')

    print(f'\ninjection: t = 0-{t_inj[-1]:.4e} s ({t_inj.size - 1} steps);  '
          f'CQ: t = 0-{t_cq[-1]:.4e} s ({t_cq.size - 1} saved steps)')
    if not os.path.isfile(EQUILIBRIUM):
        print(f'WARNING: {EQUILIBRIUM} not found; copy it there (or set EQUILIBRIUM) before running dreami.')


if __name__ == '__main__':
    main()
