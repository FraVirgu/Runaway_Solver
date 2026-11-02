#!/bin/bash
#SBATCH --job-name=dump_snap
#SBATCH --output=dump_snap_%j.out
#SBATCH --error=dump_snap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256         # reserve every core as a task slot
#SBATCH --cpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive
#
# Run dreami once on every snapshot settings file written by
# examples/kinetic_from_snapshots.py,
#
#   $DATADIR/dream_settings_kinetic_<snapshot>_L<LEVEL>.h5
#
# each in its own folder $DATADIR/sys_<snapshot>_L<LEVEL>, with
# -dream_save_system. dreami writes in the current directory, so every
# folder ends up with what the scaling scripts
# (run_strong_scaling_radius_{cores,nodes}_snapshots.sh) read:
#
#   petsc_mat_np1_step1_iter1, petsc_rhs_np1_step1_iter1,
#   petsc_solution_np1_step1_iter1 (the reference solution),
#   petsc_block_sizes.txt,
#
# plus the DREAM output file and the log dreami.log.
#
# dreami runs on 1 rank: the scaling scripts look for the "np1" files.
#
# Submit with
#
#   $ sbatch run_dump_snapshots.sh
#   $ SNAPSHOTS="CQ_tf injection_t0" sbatch run_dump_snapshots.sh
#   $ LEVEL=2 sbatch run_dump_snapshots.sh
#   $ PARALLEL=1 sbatch run_dump_snapshots.sh     # all snapshots at the same time
#
# PARALLEL=1 runs the (independent, serial) dreami of every snapshot
# concurrently on the node: much faster, as long as they fit in memory
# together.

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

export PETSC_OPTIONS=${PETSC_OPTIONS:-"-dream_kinetic_only -dream_log_rebuild 3  -dream_log_matrix -dream_scale "diag""}
export PETSC_OPTIONS="$PETSC_OPTIONS -dream_save_system"

LEVEL=${LEVEL:-4}
PARALLEL=${PARALLEL:-0}

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
EXE=$DREAM_ROOT/build/iface/dreami
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}

if [ ! -f $EXE ]; then
    echo "'$EXE' not found." >&2
    exit 1
fi

# all snapshots with a settings file at this LEVEL, unless SNAPSHOTS is given
if [ -z "$SNAPSHOTS" ]; then
    for f in $DATADIR/dream_settings_kinetic_*_L${LEVEL}.h5; do
        [ -f "$f" ] || continue
        s=$(basename $f .h5)
        s=${s#dream_settings_kinetic_}
        SNAPSHOTS="$SNAPSHOTS ${s%_L${LEVEL}}"
    done
fi
if [ -z "$SNAPSHOTS" ]; then
    echo "No dream_settings_kinetic_*_L${LEVEL}.h5 in $DATADIR: run examples/kinetic_from_snapshots.py first." >&2
    exit 1
fi
for snap in $SNAPSHOTS; do
    if [ ! -f $DATADIR/dream_settings_kinetic_${snap}_L${LEVEL}.h5 ]; then
        echo "'dream_settings_kinetic_${snap}_L${LEVEL}.h5' not found (in $DATADIR)." >&2
        exit 1
    fi
done

echo "Snapshots: $SNAPSHOTS   (LEVEL = $LEVEL, PETSC_OPTIONS = '$PETSC_OPTIONS')"

# Run dreami on one snapshot in its folder; prints the result when done
run_snapshot() {
    local snap=$1
    local dir=$DATADIR/sys_${snap}_L${LEVEL}
    mkdir -p $dir
    cd $dir
    # remove the files of a previous run, so a failed run cannot leave old ones behind
    rm -f petsc_mat_np1_step1_iter1* petsc_rhs_np1_step1_iter1* petsc_solution_np1_step1_iter1* \
          petsc_solution_final_step* petsc_block_sizes.txt

    local t0=$(date +%s)
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exact --cpu-bind=cores \
         $EXE $DATADIR/dream_settings_kinetic_${snap}_L${LEVEL}.h5 > dreami.log 2>&1
    local code=$?
    local t1=$(date +%s)

    local missing=""
    for f in petsc_mat_np1_step1_iter1 petsc_rhs_np1_step1_iter1 petsc_solution_np1_step1_iter1 petsc_block_sizes.txt; do
        [ -f $f ] || missing="$missing $f"
    done
    if (( code == 0 )) && [ -z "$missing" ]; then
        echo "=== $snap: OK in $((t1 - t0)) s, $(awk '$1 == "Nr" {print "Nr = " $2}' petsc_block_sizes.txt), files in $dir"
    else
        echo "=== $snap: FAILED (exit code $code, missing:${missing:- none}) after $((t1 - t0)) s, see $dir/dreami.log"
    fi
}

for snap in $SNAPSHOTS; do
    if (( PARALLEL )); then
        run_snapshot $snap &
    else
        run_snapshot $snap
    fi
done
wait
