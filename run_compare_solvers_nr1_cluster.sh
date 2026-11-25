#!/bin/bash
#SBATCH --job-name=cmp_nr1
#SBATCH --output=cmp_nr1_%j.out
#SBATCH --error=cmp_nr1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9           # one serial run per snapshot (PARALLEL=1)
#SBATCH --cpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=64G                     # shared node: everything here is serial
#
# Cluster version of run_compare_solvers_nr1.sh: serial comparison of
# LU (direct) vs AMG + GMRES vs ILU + GMRES on the kinetic system
# (f_hot, f_re) of the snapshots with ONE radial cell.
#
# For each snapshot, in $DATADIR/sys_toroidal_<snapshot>_nr1_L<LEVEL>:
#   0. (if missing) the settings file is written by
#      examples/kinetic_snapshot_toroidal.py with NR=1;
#   1. dreami, 1 rank, -dream_kinetic_only -dream_save_system: dumps the
#      matrix, rhs, reference solution and petsc_block_sizes.txt;
#   2. testkineticsolversserial -solvers $SOLVERS on them (1 rank).
# Results: solvers.log in each folder, all of them in
# $DATADIR/compare_solvers_nr1_L<LEVEL>.txt.
#
# Build testkineticsolversserial first (make testkineticsolversserial in
# $DREAM_ROOT/build). Submit from the Runaway_Solver folder with
#
#   $ sbatch run_compare_solvers_nr1_cluster.sh                 # LEVEL 1, all snapshots
#   $ LEVEL=2 sbatch run_compare_solvers_nr1_cluster.sh
#   $ SNAPSHOTS="CQ_t29ms CQ_tf" sbatch run_compare_solvers_nr1_cluster.sh
#   $ PARALLEL=1 sbatch run_compare_solvers_nr1_cluster.sh      # all snapshots at once
#   $ SKIP_DUMP=1 SOLVERS=lu,amg,ilu,bjamgeu sbatch run_compare_solvers_nr1_cluster.sh
#   $ EXTRA="-lu_pc_factor_mat_solver_type petsc" sbatch run_compare_solvers_nr1_cluster.sh
#     (the LU uses MUMPS by default when PETSc has it)
#
# Environment for the settings (passed to kinetic_snapshot_toroidal.py):
# NT_CQ, CQ_FULL_DT, DT_TARGET, ADV, ... (default: CQ = one step of ~1e-7 s).
#
# PARALLEL=1 runs the (independent, serial) snapshots concurrently on the
# node; the timings are then less clean (shared memory bandwidth), so use
# it for a quick look and PARALLEL=0 (default) for the numbers to keep.

PETSC_MODULE=${PETSC_MODULE:-petsc/3.25.2-intelMPI-intel-hypre}

module use /pitagora_work/FUPA2_REDISMHD/fcipolle/modulefiles
module purge
module load intel-oneapi-compilers/2024.1.0
module load intel-oneapi-mpi/2021.12.1
module load intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1
module load hdf5/1.14.3--oneapi--2024.1.0
module load gmake/4.4.1
module load $PETSC_MODULE
module load gsl/2.7.1--intel-oneapi-mpi--2021.12.1--oneapi--2024.1.0
module load cmake/3.27.9

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

LEVEL=${LEVEL:-1}
SOLVERS=${SOLVERS:-lu,amg,ilu}
SKIP_DUMP=${SKIP_DUMP:-0}
PARALLEL=${PARALLEL:-0}
EXTRA=${EXTRA:-}

SRC=${SRC:-${SLURM_SUBMIT_DIR:-$(pwd)}}          # the Runaway_Solver folder
export DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
DREAMI=$DREAM_ROOT/build/iface/dreami
TEST=$DREAM_ROOT/build/iface/testkineticsolversserial
GEN=$SRC/examples/kinetic_snapshot_toroidal.py

for exe in $DREAMI $TEST; do
    if [ ! -x $exe ]; then
        echo "'$exe' not found: build it first (make dreami testkineticsolversserial)." >&2
        exit 1
    fi
done

# settings file of a snapshot (CQ ones carry a _dt<step> tag)
settings_of() {
    ls $DATADIR/dream_settings_toroidal_${1}_nr1_L${LEVEL}.h5 \
       $DATADIR/dream_settings_toroidal_${1}_nr1_dt*_L${LEVEL}.h5 \
       $DATADIR/dream_settings_toroidal_${1}_nr1_nt*_L${LEVEL}.h5 2>/dev/null | head -1
}

# ---- settings: write the missing ones ----
if [ -z "$SNAPSHOTS" ]; then
    SNAPSHOTS=$(ls -d $SRC/DATASET/snapshots/*/ 2>/dev/null | xargs -n1 basename)
fi
if [ -z "$SNAPSHOTS" ]; then
    echo "No snapshots in $SRC/DATASET/snapshots (submit from the Runaway_Solver folder, or set SRC)." >&2
    exit 1
fi
for snap in $SNAPSHOTS; do
    if [ -z "$(settings_of $snap)" ]; then
        echo "Writing the Nr = 1 settings of $snap (LEVEL $LEVEL)"
        NR=1 LEVEL=$LEVEL SNAPSHOT=$snap KINETIC_OUT_DIR=$DATADIR python3 $GEN > /dev/null \
            || echo "  kinetic_snapshot_toroidal.py failed for $snap" >&2
    fi
done

SUMMARY=$DATADIR/compare_solvers_nr1_L${LEVEL}.txt
echo "Nr = 1, LEVEL = $LEVEL, solvers $SOLVERS  $EXTRA   (job ${SLURM_JOB_ID:-local})" | tee $SUMMARY

# ---- one snapshot: dump + compare; writes $dir/summary.txt ----
run_snapshot() {
    local snap=$1
    local settings=$(settings_of $snap)
    local dir=$DATADIR/sys_toroidal_${snap}_nr1_L${LEVEL}
    mkdir -p $dir
    cd $dir
    : > summary.txt
    if [ -z "$settings" ]; then
        echo "=== $snap: no settings file, skipped" > summary.txt
        return
    fi

    if (( ! SKIP_DUMP )); then
        rm -f petsc_mat_np1_* petsc_rhs_np1_* petsc_solution_* petsc_block_sizes.txt
        PETSC_OPTIONS="-dream_kinetic_only -dream_scale diag -dream_save_system" \
            srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exact --cpu-bind=cores \
                 $DREAMI $settings > dreami.log 2>&1
        if [ ! -f petsc_mat_np1_step1_iter1 ] || [ ! -f petsc_block_sizes.txt ]; then
            echo "=== $snap: dreami did not dump the system, see $dir/dreami.log" > summary.txt
            return
        fi
    fi

    local t0=$(date +%s)
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exact --cpu-bind=cores \
         $TEST -solvers $SOLVERS $EXTRA > solvers.log 2>&1
    local code=$?
    local t1=$(date +%s)

    {
        echo "=== $snap   ($(basename $settings))"
        grep -E "^Matrix|its +[0-9]+|reason:|inner GMRES" solvers.log
        (( code == 0 )) || echo "    testkineticsolversserial exit code $code after $((t1 - t0)) s, see $dir/solvers.log"
    } > summary.txt
    (( PARALLEL )) || cat summary.txt
}

for snap in $SNAPSHOTS; do
    if (( PARALLEL )); then
        run_snapshot $snap &
    else
        run_snapshot $snap
    fi
done
wait

# collect, in snapshot order
for snap in $SNAPSHOTS; do
    cat $DATADIR/sys_toroidal_${snap}_nr1_L${LEVEL}/summary.txt
done > $SUMMARY.body
(( PARALLEL )) && cat $SUMMARY.body
cat $SUMMARY.body >> $SUMMARY
rm -f $SUMMARY.body

echo
echo "Summary in $SUMMARY"
