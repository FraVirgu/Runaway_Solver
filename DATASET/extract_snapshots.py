#!/usr/bin/env python3
#
# Extract single-time snapshots (all unknowns) from the DREAM outputs in this
# folder, together with the dreami settings needed to run the simulations.
#
# Snapshots written to snapshots/<name>/:
#   injection_t0   first time step of output_injection_start_M4.h5
#   injection_tf   last time step of output_injection_start_M4.h5
#   CQ_tf          last time step of output_CQ_M4.h5
#
# Each snapshot folder contains:
#   state.h5      all unknowns at that time, stored in the DREAM output layout
#                 (open with DREAMOutput('state.h5'), or the DREAM cli)
#   settings.h5   settings of the simulation the snapshot comes from
#                 (re-runs that simulation from scratch)
#   restart.h5    same settings, but initialized from state.h5
#                 (starts a new simulation from this snapshot)
#
# Run dreami from inside the snapshot folder, since paths are relative:
#   $ cd snapshots/CQ_tf && dreami settings.h5
#
# The simulations use the numerical ITER equilibrium ITER_H26_LUKE.h5, which
# is not part of the release: copy it into this folder (DATASET/) before
# running dreami.
#
# Usage:
#   $ python extract_snapshots.py

import os
import sys

import h5py

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', os.path.join(DATASET_DIR, '..', '..', 'DREAM')))
sys.path.append(os.path.join(DREAM_ROOT, 'py'))

from DREAM.DREAMSettings import DREAMSettings
import DREAM.DREAMIO as DREAMIO

INJECTION = 'output_injection_start_M4.h5'
CQ = 'output_CQ_M4.h5'
EQUILIBRIUM = 'ITER_H26_LUKE.h5'
SNAP_DIR = os.path.join(DATASET_DIR, 'snapshots')

# name -> (source output file, time index)
SNAPSHOTS = {
    'injection_t0': (INJECTION, 0),
    'injection_tf': (INJECTION, -1),
    'CQ_tf':        (CQ, -1),
}

# Groups copied unchanged into the snapshot (time-independent)
STATIC_GROUPS = ['code', 'ionmeta', 'settings']


def write_state(src_file, timeindex, dst_file):
    """Write all unknowns at one time index to a DREAM-like output file."""
    with h5py.File(src_file, 'r') as src, h5py.File(dst_file, 'w') as dst:
        nt = src['grid/t'].shape[0]
        k = timeindex % nt

        for g in STATIC_GROUPS:
            if g in src:
                src.copy(src[g], dst)

        # Grid: everything as is, except the time vector
        grid = dst.create_group('grid')
        for key, val in src['grid'].items():
            if key != 't':
                src.copy(val, grid)
        grid.create_dataset('t', data=src['grid/t'][k:k+1])

        # Unknowns: keep only time index k (first dimension), with attributes
        eqsys = dst.create_group('eqsys')
        for name, ds in src['eqsys'].items():
            out = eqsys.create_dataset(name, data=ds[k:k+1])
            for a, v in ds.attrs.items():
                out.attrs[a] = v

        return k, src['grid/t'][k], len(src['eqsys'])


def load_settings(src_file, folder):
    """Settings stored inside a DREAM output file (without loading the data)."""
    ds = DREAMSettings(src_file, path='settings')
    # Look for the equilibrium file in DATASET/, relative to the snapshot folder
    ds.radialgrid.num_filename = os.path.relpath(os.path.join(DATASET_DIR, EQUILIBRIUM), folder)
    return ds


def save_settings(ds, filename):
    # No verification: it fails while the equilibrium file is missing
    DREAMIO.SaveDictAsHDF5(filename, ds.todict(verify=False))


def run_settings(src_file, folder):
    """Settings that re-run the simulation which produced 'src_file'."""
    ds = load_settings(src_file, folder)
    if src_file.endswith(CQ):
        # The CQ simulation starts from the end of the injection simulation
        ds.fromOutput(os.path.relpath(os.path.join(DATASET_DIR, INJECTION), folder), relpath=True, timeindex=-1)
    ds.output.setFilename(os.path.basename(src_file))
    return ds


def restart_settings(src_file, folder, name):
    """Settings that start a new simulation from the snapshot state.h5."""
    ds = load_settings(src_file, folder)
    ds.fromOutput('state.h5', relpath=True, timeindex=0)
    ds.output.setFilename(f'output_restart_{name}.h5')
    return ds


def main():
    for name, (fname, timeindex) in SNAPSHOTS.items():
        src_file = os.path.join(DATASET_DIR, fname)
        folder = os.path.join(SNAP_DIR, name)
        os.makedirs(folder, exist_ok=True)

        k, t, nunknowns = write_state(src_file, timeindex, os.path.join(folder, 'state.h5'))
        save_settings(run_settings(src_file, folder), os.path.join(folder, 'settings.h5'))
        save_settings(restart_settings(src_file, folder, name), os.path.join(folder, 'restart.h5'))

        print(f'{name:14s} {fname} index {k:5d}  t = {t:.6e} s  ({nunknowns} unknowns) -> {folder}')

    if not os.path.isfile(os.path.join(DATASET_DIR, EQUILIBRIUM)):
        print(f'\nWARNING: {EQUILIBRIUM} not found in {DATASET_DIR}; copy it there before running dreami.')


if __name__ == '__main__':
    main()
