#!/bin/bash
#
# Local, serial comparison of LU (direct) vs AMG + GMRES vs ILU + GMRES on the
# kinetic system (f_hot, f_re) of the snapshots with ONE radial cell.
#
# Settings: examples/kinetic_snapshot_toroidal.py with NR=1, e.g.
#
#   $ NR=1 LEVEL=1 python3 examples/kinetic_snapshot_toroidal.py
#
# which writes $DATADIR/dream_settings_toroidal_<snapshot>_nr1[_dt1e-07]_L<LEVEL>.h5.
# For each snapshot this script
#   1. runs dreami (1 rank, -dream_kinetic_only -dream_save_system) in
#      $DATADIR/sys_toroidal_<snapshot>_nr1_L<LEVEL>, which leaves the matrix,
#      rhs, reference solution and petsc_block_sizes.txt there;
#   2. runs testkineticsolversserial -solvers $SOLVERS on them.
# Results: one line per solver in that folder's solvers.log, and all of them
# in $DATADIR/compare_solvers_nr1_L<LEVEL>.txt.
#
#   $ ./run_compare_solvers_nr1.sh
#   $ SNAPSHOTS="CQ_t29ms CQ_tf" ./run_compare_solvers_nr1.sh
#   $ SKIP_DUMP=1 ./run_compare_solvers_nr1.sh      # reuse the dumped systems
#   $ EXTRA="-ilu_pc_factor_levels 1" ./run_compare_solvers_nr1.sh
#
# Local machine: keep LEVEL <= 1 (L2+ goes to the cluster).

LEVEL=${LEVEL:-1}
SOLVERS=${SOLVERS:-lu,amg,ilu}
SKIP_DUMP=${SKIP_DUMP:-0}
EXTRA=${EXTRA:-}

DREAM_ROOT=${DREAM_ROOT:-$HOME/Desktop/DREAM}
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
DREAMI=$DREAM_ROOT/build/iface/dreami
TEST=$DREAM_ROOT/build/iface/testkineticsolversserial

for exe in $DREAMI $TEST; do
    if [ ! -x $exe ]; then
        echo "'$exe' not found: build it first (make testkineticsolversserial dreami)." >&2
        exit 1
    fi
done

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# settings file of a snapshot (CQ ones carry a _dt<step> tag)
settings_of() {
    ls $DATADIR/dream_settings_toroidal_${1}_nr1_L${LEVEL}.h5 \
       $DATADIR/dream_settings_toroidal_${1}_nr1_dt*_L${LEVEL}.h5 2>/dev/null | head -1
}

if [ -z "$SNAPSHOTS" ]; then
    for f in $DATADIR/dream_settings_toroidal_*_nr1*_L${LEVEL}.h5; do
        [ -f "$f" ] || continue
        s=$(basename $f .h5)
        s=${s#dream_settings_toroidal_}
        SNAPSHOTS="$SNAPSHOTS ${s%%_nr1*}"
    done
fi
if [ -z "$SNAPSHOTS" ]; then
    echo "No dream_settings_toroidal_*_nr1*_L${LEVEL}.h5 in $DATADIR: run" >&2
    echo "  NR=1 LEVEL=$LEVEL python3 examples/kinetic_snapshot_toroidal.py" >&2
    exit 1
fi

SUMMARY=$DATADIR/compare_solvers_nr1_L${LEVEL}.txt
echo "Nr = 1, LEVEL = $LEVEL, solvers $SOLVERS  $EXTRA" | tee $SUMMARY

for snap in $SNAPSHOTS; do
    settings=$(settings_of $snap)
    if [ -z "$settings" ]; then
        echo "=== $snap: no settings file, skipped" | tee -a $SUMMARY
        continue
    fi
    dir=$DATADIR/sys_toroidal_${snap}_nr1_L${LEVEL}
    mkdir -p $dir
    cd $dir

    if (( ! SKIP_DUMP )); then
        rm -f petsc_mat_np1_* petsc_rhs_np1_* petsc_solution_* petsc_block_sizes.txt
        PETSC_OPTIONS="-dream_kinetic_only -dream_scale diag -dream_save_system" \
            mpirun -n 1 $DREAMI $settings > dreami.log 2>&1
        if [ ! -f petsc_mat_np1_step1_iter1 ] || [ ! -f petsc_block_sizes.txt ]; then
            echo "=== $snap: dreami did not dump the system, see $dir/dreami.log" | tee -a $SUMMARY
            continue
        fi
    fi

    echo "=== $snap   ($(basename $settings))" | tee -a $SUMMARY
    mpirun -n 1 $TEST -solvers $SOLVERS $EXTRA > solvers.log 2>&1
    # matrix size line + one line per solver (and failure reasons)
    grep -E "^Matrix|its +[0-9]+|reason:|inner GMRES" solvers.log | tee -a $SUMMARY
done

echo
echo "Summary in $SUMMARY"
