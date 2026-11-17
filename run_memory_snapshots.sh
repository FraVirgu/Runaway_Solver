#!/bin/bash
#SBATCH --job-name=mem_snap
#SBATCH --output=mem_snap_%j.out
#SBATCH --error=mem_snap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256         # reserve every core as a task slot, so steps up to 256 ranks fit
#SBATCH --cpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # the node to ourselves: nothing else in the RSS figures
#
# Memory scaling of the kinetic-only code with the number of MPI ranks, on
# the dataset snapshots: for every snapshot in SNAPSHOTS and every n in
# NPROCS, on ONE node, it runs
#
#   test    testcollisionquantity (iface/TestCollisionQuantity.cpp): builds
#           the simulation and does the first rebuild, then reports per
#           rank the bytes of the distributed data (collision quantities,
#           transient weights, interpolation and advection/diffusion
#           coefficients, PETSc matrix) and the RSS at every stage of the
#           construction -> memory.txt
#   dreami  the whole run with -dream_log_memory: RSS at every stage of the
#           construction, of the first time step (rebuild, matrix, solve
#           including the preconditioner setup) and after all the time
#           steps -> dreami.log
#
# on $DATADIR/dream_settings_kinetic_<snapshot>_L<LEVEL>.h5
# (examples/kinetic_from_snapshots.py). Every run has its own folder
#
#   $OUTDIR/<snapshot>/<test|dreami>_np<n>/
#
# and summarize_memory.py turns the whole $OUTDIR into tables (and a plot):
#
#   $ python3 summarize_memory.py $OUTDIR
#
# Submit with
#
#   $ sbatch run_memory_snapshots.sh                          # L2, all snapshots, test + dreami
#   $ SNAPSHOTS="CQ_tf injection_t0" sbatch run_memory_snapshots.sh
#   $ NPROCS="1 11 22" RUNS=test sbatch run_memory_snapshots.sh
#   $ LEVEL=4 sbatch run_memory_snapshots.sh
#
# The radial block Jacobi needs n <= Nr or n a multiple of Nr (each radius
# split over n/Nr ranks): other n > Nr are skipped. n < Nr that do not divide
# Nr run, with unequal radii per rank (the max-per-rank figures then show the
# rank with the most radii). With Nr = 11: NPROCS="1 11 22 44 88 176".
#
# The memory figures are per process (resident set size): a fixed part
# (MPI, PETSc, hypre, the executable, ADAS...) does not shrink with n, which
# summarize_memory.py reports apart.

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

# Same options as run_dump_snapshots.sh, without -dream_save_system (writing
# the system would only add I/O buffers to the figures)
export PETSC_OPTIONS=${PETSC_OPTIONS:-"-dream_kinetic_only -dream_scale diag"}

LEVEL=${LEVEL:-2}
NPROCS=${NPROCS:-"1 11 22 44 88 176"}
RUNS=${RUNS:-"test dreami"}           # which executables to run

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=${OUTDIR:-$DATADIR/memory_L${LEVEL}_${SLURM_JOB_ID:-local}}

CORES_PER_NODE=256                    # 2 sockets x 128 cores

for run in $RUNS; do
    case $run in
        test)   exe=$DREAM_ROOT/build/iface/testcollisionquantity ;;
        dreami) exe=$DREAM_ROOT/build/iface/dreami ;;
        *)      echo "Unknown run '$run' in RUNS (test, dreami)." >&2; exit 1 ;;
    esac
    if [ ! -x $exe ]; then
        echo "'$exe' not found: build it first (make $(basename $exe))." >&2
        exit 1
    fi
done

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
# check every file first: do not find out about a missing one hours into the job
for snap in $SNAPSHOTS; do
    if [ ! -f $DATADIR/dream_settings_kinetic_${snap}_L${LEVEL}.h5 ]; then
        echo "'dream_settings_kinetic_${snap}_L${LEVEL}.h5' not found (in $DATADIR)." >&2
        exit 1
    fi
done

mkdir -p $OUTDIR
echo "Snapshots: $SNAPSHOTS"
echo "LEVEL = $LEVEL, NPROCS = '$NPROCS', RUNS = '$RUNS'"
echo "PETSC_OPTIONS = '$PETSC_OPTIONS'"
echo "Outputs in $OUTDIR"

for snap in $SNAPSHOTS; do
    SETTINGS=$DATADIR/dream_settings_kinetic_${snap}_L${LEVEL}.h5
    # Nr = number of radial cells (r_f holds the Nr+1 cell faces)
    NR=$(python3 -c "import h5py; print(h5py.File('$SETTINGS', 'r')['radialgrid/r_f'].shape[0] - 1)" 2>/dev/null)
    echo
    echo "##### $snap: Nr = ${NR:-unknown} #####"

    for n in $NPROCS; do
        if (( n > CORES_PER_NODE )); then
            echo "skipping n = $n: more than the $CORES_PER_NODE cores of the node"
            continue
        fi
        if [ -n "$NR" ] && (( n > NR && n % NR != 0 )); then
            echo "skipping n = $n: more ranks than Nr = $NR and not a multiple of it (the radial block Jacobi refuses it)"
            continue
        fi
        if [ -n "$NR" ] && (( n < NR && NR % n != 0 )); then
            echo "warning: n = $n does not divide Nr = $NR, the ranks get unequal numbers of radii"
        fi
        # spread the ranks over the whole node (both sockets, memory channels)
        CPUS_PER_TASK=$((CORES_PER_NODE / n))

        for run in $RUNS; do
            dir=$OUTDIR/$snap/${run}_np${n}
            mkdir -p $dir
            cd $dir

            t0=$(date +%s)
            if [ $run == test ]; then
                srun --nodes=1 --ntasks=$n --ntasks-per-node=$n --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores \
                     $DREAM_ROOT/build/iface/testcollisionquantity $SETTINGS -cq_dir $dir > test.log 2>&1
                code=$?
                ok=$([ -f cq_np${n}_memory.txt ] && grep -q "End of resident set size table" cq_np${n}_memory.txt && echo 1)
                log=test.log
            else
                # dreami writes its output file in the current directory
                srun --nodes=1 --ntasks=$n --ntasks-per-node=$n --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores \
                     $DREAM_ROOT/build/iface/dreami $SETTINGS -dream_log_memory > dreami.log 2>&1
                code=$?
                # dreami returns 0 even when the run failed: the table is
                # printed only after a complete run
                ok=$(grep -q "End of resident set size table" dreami.log && echo 1)
                log=dreami.log
            fi
            t1=$(date +%s)

            if [ -n "$ok" ]; then
                echo "=== $snap $run np$n: OK in $((t1 - t0)) s"
            else
                echo "=== $snap $run np$n: FAILED (exit code $code) after $((t1 - t0)) s, see $dir/$log"
            fi
        done
    done
done

# summarize_memory.py next to this script (under sbatch $0 is a copy in the
# spool directory, so look in the submission directory)
SUMMARY=${SUMMARY:-${SLURM_SUBMIT_DIR:-$(dirname $(readlink -f $0))}/summarize_memory.py}
echo
if [ -f $SUMMARY ]; then
    python3 $SUMMARY $OUTDIR --plot $OUTDIR/memory_scaling.png
else
    echo "'$SUMMARY' not found: run 'python3 summarize_memory.py $OUTDIR' yourself."
fi
