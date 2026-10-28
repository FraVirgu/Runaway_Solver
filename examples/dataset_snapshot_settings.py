#!/usr/bin/env python3
#
# Create dreami settings files for every snapshot in DATASET/snapshots/
# (written by DATASET/extract_snapshots.py) and put them in the dreami
# folder DREAM/build/iface.
#
# For each snapshot <name> two settings files are written:
#
#   dream_settings_<name>.h5        new simulation initialized from the
#                                   snapshot state (snapshots/<name>/state.h5)
#   dream_settings_<name>_rerun.h5  the simulation the snapshot comes from,
#                                   run again from its own initial condition
#
# All file paths inside the settings (initial state, ITER equilibrium) are
# made absolute, so dreami can be started from build/iface.
#
# Run as
#
#   $ ./dataset_snapshot_settings.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_CQ_tf.h5
#
# ###################################################################

import glob
import os
import sys
import numpy as np

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', os.path.join(EXAMPLES_DIR, '..', '..', 'DREAM')))
OUT_DIR = os.path.expanduser(
    os.environ.get('SNAPSHOT_OUT_DIR', os.path.join(DREAM_ROOT, 'build', 'iface')))
SNAP_DIR = os.path.join(EXAMPLES_DIR, '..', 'DATASET', 'snapshots')

sys.path.append(os.path.join(DREAM_ROOT, 'py'))

from DREAM.DREAMSettings import DREAMSettings
import DREAM.DREAMIO as DREAMIO

# settings file in the snapshot folder -> suffix of the generated file
KINDS = {
    'restart.h5':  '',
    'settings.h5': '_rerun',
}


def absolute(path, folder):
    """Resolve a path stored relative to the snapshot folder."""
    return os.path.normpath(os.path.join(folder, path))


def make_settings(folder, kind, tag):
    ds = DREAMSettings(os.path.join(folder, kind))

    if ds.init.get('fromfile'):
        ds.fromOutput(absolute(ds.init['fromfile'], folder),
                      ignore=ds.init.get('eqsysignore', []),
                      timeindex=int(np.ravel(ds.init.get('filetimeindex', -1))[0]))

    eq_file = absolute(ds.radialgrid.num_filename, folder)
    ds.radialgrid.num_filename = eq_file
    ds.output.setFilename(f'output_{tag}.h5')

    settings_file = os.path.join(OUT_DIR, f'dream_settings_{tag}.h5')
    if os.path.isfile(eq_file):
        ds.save(settings_file)
    else:
        # Verification fails while the equilibrium file is missing
        DREAMIO.SaveDictAsHDF5(settings_file, ds.todict(verify=False))
    return settings_file, eq_file


folders = sorted(f for f in glob.glob(os.path.join(SNAP_DIR, '*')) if os.path.isdir(f))
if not folders:
    sys.exit(f'No snapshots found in {os.path.normpath(SNAP_DIR)}; run DATASET/extract_snapshots.py first.')

os.makedirs(OUT_DIR, exist_ok=True)
missing = set()
for folder in folders:
    name = os.path.basename(folder)
    for kind, suffix in KINDS.items():
        if not os.path.isfile(os.path.join(folder, kind)):
            continue
        settings_file, eq_file = make_settings(folder, kind, name + suffix)
        if not os.path.isfile(eq_file):
            missing.add(eq_file)
        print(f'Wrote {settings_file}')

print()
print(f'Run:  cd {OUT_DIR} && ./dreami dream_settings_<name>.h5')
for eq_file in missing:
    print(f'WARNING: equilibrium file {eq_file} not found; copy it there before running dreami.')
