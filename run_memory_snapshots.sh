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
#           steps, and the timings (time loop, rebuild, matrix, invert,
#           first and second solve; the settings must have
#           output.setTiming(stdout=True), as kinetic_from_snapshots.py
#           does) -> dreami.log, wall time of srun -> wall_s.txt
#
# on $DATADIR/dream_settings_${PREFIX}_<snapshot>_L<LEVEL>.h5
# (examples/kinetic_from_snapshots.py). Every run has its own folder
#
#   $OUTDIR/<snapshot>/<test|dreami>_np<n>/          (first run)
#   $OUTDIR/<snapshot>/dreami_np<n>_r<k>/            (repeat k >= 2)
#
# dreami is run REPEATS times (default 1) for less noisy timings; the
# memory figures are taken from the first run, the timings from the fastest.
# summarize_memory.py turns the whole $OUTDIR into tables (and plots):
#
#   $ python3 summarize_memory.py $OUTDIR
#
# Submit with
#
#   $ sbatch run_memory_snapshots.sh                          # L2, all snapshots, test + dreami
#   $ SNAPSHOTS="CQ_tf injection_t0" sbatch run_memory_snapshots.sh
#   $ NPROCS="1 11 22" RUNS=test sbatch run_memory_snapshots.sh
#   $ RUNS=dreami REPEATS=3 sbatch run_memory_snapshots.sh    # speed-up only, 3 runs each
#   $ LEVEL=4 sbatch run_memory_snapshots.sh
#   $ PREFIX=toroidal SNAPSHOTS="CQ_tf_nr44" NPROCS="1 11 22 44" sbatch run_memory_snapshots.sh
#                                                             # toroidal geometry, 44 radii
#
# Weak scaling (one radius per rank): NPROCS=nr runs every snapshot with n =
# its Nr. The settings come from kinetic_from_snapshots.py with NR set
# (snapshot names <name>_nr<Nr>); summarize_memory.py groups the snapshots
# with the same <name> and prints the weak-scaling table:
#
#   $ for nr in 11 22 44 88 176; do SNAPSHOT=CQ_tf NR=$nr examples/kinetic_from_snapshots.py; done
#   $ SNAPSHOTS="CQ_tf_nr11 CQ_tf_nr22 CQ_tf_nr44 CQ_tf_nr88 CQ_tf_nr176" NPROCS=nr \
#         RUNS=dreami REPEATS=3 sbatch run_memory_snapshots.sh
#
# Strong scaling at Nr = 44 (divisors and multiples of 44):
#
#   $ SNAPSHOTS="CQ_tf_nr44" NPROCS="1 2 4 11 22 44 88 176" REPEATS=3 sbatch run_memory_snapshots.sh
#
# "nr" can be mixed with numbers (NPROCS="1 nr"): the same n is run once.
#
# Several nodes: submit with sbatch --nodes=N. Every n runs on nn <= N nodes
# with ppn = n/nn ranks per node, spread over all the cores of a node:
#   RANKS_PER_NODE unset   nn = the fewest nodes with n/nn <= 256 and nn | n
#                          (n <= 256 stays on 1 node, as before)
#   RANKS_PER_NODE=p       nn = n/p (n <= p: 1 node): a fixed density, so
#                          larger n means more nodes, not fuller nodes
# n that do not fit are skipped. The node count of every run is in its
# nodes.txt and in the summary. Nr = 44 on up to 4 nodes (704 ranks = each
# radius split over 16 ranks):
#
#   $ SNAPSHOTS="CQ_tf_nr44" NPROCS="44 88 176 352 704" RANKS_PER_NODE=176 REPEATS=3 \
#         sbatch --nodes=4 run_memory_snapshots.sh            # 1,1,1,2,4 nodes
#   $ SNAPSHOTS="CQ_tf_nr44" NPROCS="44 88 176" RANKS_PER_NODE=44 REPEATS=3 \
#         sbatch --nodes=4 run_memory_snapshots.sh            # 1,2,4 nodes x 44 ranks
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
# settings files dream_settings_<PREFIX>_<snapshot>_L<LEVEL>.h5:
#   kinetic   examples/kinetic_from_snapshots.py (cylindrical; <snapshot> = <name>_nr<NR> with NR set)
#   toroidal  examples/kinetic_snapshot_toroidal.py (<snapshot> = <name>_nr<NR>)
PREFIX=${PREFIX:-kinetic}
NPROCS=${NPROCS:-"1 11 22 44 88 176"}
RUNS=${RUNS:-"test dreami"}           # which executables to run
REPEATS=${REPEATS:-1}                 # runs of dreami per (snapshot, n), for the timings

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=${OUTDIR:-$DATADIR/memory_${PREFIX}_L${LEVEL}_${SLURM_JOB_ID:-local}}

CORES_PER_NODE=256                    # 2 sockets x 128 cores
JOB_NODES=${SLURM_JOB_NUM_NODES:-1}   # sbatch --nodes=N
RANKS_PER_NODE=${RANKS_PER_NODE:-}    # empty: as few nodes as possible (see "Several nodes")

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
    for f in $DATADIR/dream_settings_${PREFIX}_*_L${LEVEL}.h5; do
        [ -f "$f" ] || continue
        s=$(basename $f .h5)
        s=${s#dream_settings_${PREFIX}_}
        SNAPSHOTS="$SNAPSHOTS ${s%_L${LEVEL}}"
    done
fi
if [ -z "$SNAPSHOTS" ]; then
    echo "No dream_settings_${PREFIX}_*_L${LEVEL}.h5 in $DATADIR: run examples/kinetic_from_snapshots.py (PREFIX=kinetic) or kinetic_snapshot_toroidal.py (PREFIX=toroidal) first." >&2
    exit 1
fi
# check every file first: do not find out about a missing one hours into the job
for snap in $SNAPSHOTS; do
    if [ ! -f $DATADIR/dream_settings_${PREFIX}_${snap}_L${LEVEL}.h5 ]; then
        echo "'dream_settings_${PREFIX}_${snap}_L${LEVEL}.h5' not found (in $DATADIR)." >&2
        exit 1
    fi
done

mkdir -p $OUTDIR
echo "Snapshots: $SNAPSHOTS"
echo "LEVEL = $LEVEL, NPROCS = '$NPROCS', RUNS = '$RUNS', REPEATS (dreami) = $REPEATS"
echo "Nodes in the job: $JOB_NODES, RANKS_PER_NODE = ${RANKS_PER_NODE:-auto}"
echo "PETSC_OPTIONS = '$PETSC_OPTIONS'"
echo "Outputs in $OUTDIR"

for snap in $SNAPSHOTS; do
    SETTINGS=$DATADIR/dream_settings_${PREFIX}_${snap}_L${LEVEL}.h5
    # Nr = number of radial cells (r_f holds the Nr+1 cell faces); h5ls
    # (hdf5 module) prints "r_f  Dataset {12}" or "{12/12}"; h5py as fallback
    NF=$(h5ls $SETTINGS/radialgrid/r_f 2>/dev/null | sed -nE 's/.*Dataset \{([0-9]+).*/\1/p')
    [ -z "$NF" ] && NF=$(python3 -c "import h5py; print(h5py.File('$SETTINGS', 'r')['radialgrid/r_f'].shape[0])" 2>/dev/null)
    NR=$([ -n "$NF" ] && echo $((NF - 1)))
    echo
    echo "##### $snap: Nr = ${NR:-unknown} #####"

    # "nr" in NPROCS stands for this snapshot's Nr (weak scaling)
    NPROCS_SNAP=""
    for n in $NPROCS; do
        if [ "$n" == nr ]; then
            if [ -z "$NR" ]; then
                echo "skipping n = nr: Nr of $snap unknown"
                continue
            fi
            n=$NR
        fi
        [[ " $NPROCS_SNAP " == *" $n "* ]] || NPROCS_SNAP="$NPROCS_SNAP $n"
    done

    for n in $NPROCS_SNAP; do
        # nodes nn and ranks per node ppn of this run (see "Several nodes" above)
        nn=""
        if [ -n "$RANKS_PER_NODE" ]; then
            if (( n <= RANKS_PER_NODE )); then
                nn=1
            elif (( n % RANKS_PER_NODE == 0 )); then
                nn=$((n / RANKS_PER_NODE))
            fi
        else
            for ((k = 1; k <= JOB_NODES; k++)); do
                if (( n % k == 0 && n / k <= CORES_PER_NODE )); then nn=$k; break; fi
            done
        fi
        if [ -z "$nn" ] || (( nn > JOB_NODES )); then
            echo "skipping n = $n: does not fit evenly on the $JOB_NODES node(s) of the job (RANKS_PER_NODE = ${RANKS_PER_NODE:-auto})"
            continue
        fi
        ppn=$((n / nn))
        if (( ppn > CORES_PER_NODE )); then
            echo "skipping n = $n: $ppn ranks per node > $CORES_PER_NODE cores"
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
        CPUS_PER_TASK=$((CORES_PER_NODE / ppn))
        SRUN_LAYOUT="--nodes=$nn --ntasks=$n --ntasks-per-node=$ppn --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores"
        echo "--- n = $n: $nn node(s) x $ppn ranks, $CPUS_PER_TASK cores per rank"

        for run in $RUNS; do
          nrep=$([ $run == dreami ] && echo $REPEATS || echo 1)
          for r in $(seq 1 $nrep); do
            dir=$OUTDIR/$snap/${run}_np${n}$([ $r -gt 1 ] && echo _r$r)
            mkdir -p $dir
            cd $dir
            echo $nn > nodes.txt

            t0=$(date +%s.%N)
            if [ $run == test ]; then
                srun $SRUN_LAYOUT \
                     $DREAM_ROOT/build/iface/testcollisionquantity $SETTINGS -cq_dir $dir > test.log 2>&1
                code=$?
                ok=$([ -f cq_np${n}_memory.txt ] && grep -q "End of resident set size table" cq_np${n}_memory.txt && echo 1)
                log=test.log
            else
                # dreami writes its output file in the current directory.
                # -dream_log_memory goes through PETSC_OPTIONS: dreami's own
                # command line (getopt) only takes -a -h -l -s and exits on
                # anything else
                PETSC_OPTIONS="$PETSC_OPTIONS -dream_log_memory" \
                srun $SRUN_LAYOUT \
                     $DREAM_ROOT/build/iface/dreami $SETTINGS > dreami.log 2>&1
                code=$?
                # dreami returns 0 even when the run failed: the table is
                # printed only after a complete run
                ok=$(grep -q "End of resident set size table" dreami.log && echo 1)
                log=dreami.log
            fi
            t1=$(date +%s.%N)
            wall=$(echo "$t1 - $t0" | bc -l)
            echo $wall > wall_s.txt

            if [ -n "$ok" ]; then
                echo "=== $snap $run np$n run $r: OK in $wall s"
            else
                echo "=== $snap $run np$n run $r: FAILED (exit code $code) after $wall s, see $dir/$log"
            fi
          done
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
