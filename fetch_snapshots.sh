#!/bin/bash
#
# Copy the solution files (petsc_solution*, petsc_block_sizes.txt and the
# DREAM output output_*.h5) of the dumped snapshot systems (folders
# sys_<snapshot>_L<LEVEL>, written by run_dump_snapshots.sh) from the
# cluster to OUT_DATA/ of this folder, keeping the folder structure.
# The matrices and right-hand sides are NOT copied.
# Run it on your own computer.
#
#   $ ./fetch_snapshots.sh                        # all snapshots, LEVEL=2
#   $ LEVEL=4 ./fetch_snapshots.sh
#   $ SNAPSHOTS="CQ_tf injection_tf" ./fetch_snapshots.sh
#   $ LEVEL="1 2 4" ./fetch_snapshots.sh          # several levels
#
# One rsync call, so you log in only once.

CLUSTER_USER=${CLUSTER_USER:-fvirgult}
CLUSTER_HOST=${CLUSTER_HOST:-login.pitagora.cineca.it}
REMOTE_DIR=${REMOTE_DIR:-/pitagora/home/userexternal/fvirgult/DREAM/build/iface}   # $DATADIR of run_dump_snapshots.sh
LEVEL=${LEVEL:-2}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_DIR=${LOCAL_DIR:-$HERE/OUT_DATA}
mkdir -p "$LOCAL_DIR"

# rsync filters: enter the wanted sys_* folders, take only petsc_solution*
set -f                  # keep sys_* as a pattern, do not expand it locally
FILTERS=()
for lv in $LEVEL; do
    if [ -z "$SNAPSHOTS" ]; then
        dirs="sys_*_L${lv}"
    else
        dirs=""
        for snap in $SNAPSHOTS; do dirs="$dirs sys_${snap}_L${lv}"; done
    fi
    for d in $dirs; do
        FILTERS+=(--include="$d/" --include="$d/petsc_solution*"
                  --include="$d/petsc_block_sizes.txt" --include="$d/output_*.h5")
    done
done
FILTERS+=(--exclude='*')

echo "From $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR  (LEVEL = $LEVEL${SNAPSHOTS:+, SNAPSHOTS = $SNAPSHOTS})"
echo "To   $LOCAL_DIR"

rsync -av --prune-empty-dirs "${FILTERS[@]}" \
      "$CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
