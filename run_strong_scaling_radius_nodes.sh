#!/bin/bash
#SBATCH --job-name=radius_nodes
#SBATCH --output=radius_nodes_%j.out
#SBATCH --error=radius_nodes_%j.err
#SBATCH --nodes=2                     # overridden by sbatch --nodes=N
#SBATCH --time=0-02:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node
#
# Scaling of testkinetichypreradius (iface/TestKineticHypreRadius.cpp) over
# the number of NODES, one job per node count, in one of two modes:
#
#  * strong scaling: FIXED number of ranks per node (RANKS_PER_NODE), so the
#    total number of ranks grows with the nodes
#
#   $ for N in 1 2 4 8; do RANKS_PER_NODE=64 sbatch --nodes=$N run_strong_scaling_radius_nodes.sh; done
#
#  * placement: FIXED total number of MPI ranks (NTASKS, default 128), spread
#    evenly over the nodes of the job (used when RANKS_PER_NODE is not set)
#
#   $ for N in 1 2 4 8 16 32 64 128; do NTASKS=128 sbatch --nodes=$N run_strong_scaling_radius_nodes.sh; done
#
# All node counts of one sweep write to the same directory
# $DATADIR/scaling_radius_nodes_ppn${RANKS_PER_NODE} (strong scaling) or
# $DATADIR/scaling_radius_nodes_n${NTASKS} (placement): one log per (nodes,
# run) and one timings_N<nodes>.csv per job, with a row per (run, solver)
# built from the test's "its ... setup ... solve ..." lines. Merge them with
#
#   $ cd $DATADIR/scaling_radius_nodes_ppn64 && head -1 timings_N1.csv > timings.csv && tail -q -n +2 timings_N*.csv >> timings.csv
#
# The system must already be dumped in $DATADIR by dreami (save_sys on), see
# run_strong_scaling_radius_cores.sh. The radial block-Jacobi part needs the
# number of ranks to be a divisor of Nr (whole radii per rank) or a multiple
# of it (each radius split over several ranks); ASM only runs in the first case.

module use /pitagora_work/FUPA2_REDISMHD/fcipolle/modulefiles
module purge
module load intel-oneapi-compilers/2024.1.0
module load intel-oneapi-mpi/2021.12.1
module load intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1
module load hdf5/1.14.3--oneapi--2024.1.0
module load gmake/4.4.1
module load petsc/3.25.2-intelMPI-intel-hypre
module load gsl/2.7.1--intel-oneapi-mpi--2021.12.1--oneapi--2024.1.0
module load cmake/3.27.9

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

export PETSC_OPTIONS=${PETSC_OPTIONS:-"-scale diag"}

# diagnostics: Intel MPI prints the fabric/provider in use and the rank pinning
export I_MPI_DEBUG=5

CORES_PER_NODE=256                    # 2 sockets x 128 cores, 1 thread per core
REPEATS=${REPEATS:-2}

NODES=$SLURM_JOB_NUM_NODES
if [ -n "$RANKS_PER_NODE" ]; then
    # strong scaling: fixed ranks per node, total grows with the nodes
    NTASKS=$((RANKS_PER_NODE * NODES))
    SWEEP=ppn${RANKS_PER_NODE}
else
    # placement: fixed total ranks, spread over the nodes
    NTASKS=${NTASKS:-128}
    if (( NTASKS % NODES != 0 )); then
        echo "error: NTASKS=$NTASKS is not divisible by $NODES nodes" >&2
        exit 1
    fi
    RANKS_PER_NODE=$((NTASKS / NODES))
    SWEEP=n${NTASKS}
fi
if (( RANKS_PER_NODE > CORES_PER_NODE )); then
    echo "error: $RANKS_PER_NODE ranks per node > $CORES_PER_NODE cores" >&2
    exit 1
fi
# spread the ranks over all cores of the node (both sockets)
CPUS_PER_TASK=$((CORES_PER_NODE / RANKS_PER_NODE))

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
EXE=$DREAM_ROOT/build/iface/testkinetichypreradius
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=$DATADIR/scaling_radius_nodes_${SWEEP}   # all node counts of one sweep in the same directory
mkdir -p $OUTDIR
cd $DATADIR

for f in $EXE petsc_mat_np1_step1_iter1 petsc_rhs_np1_step1_iter1 petsc_block_sizes.txt; do
    if [ ! -f $f ]; then
        echo "'$f' not found (in $DATADIR)." >&2
        exit 1
    fi
done

NR=$(awk '$1 == "Nr" {print $2}' petsc_block_sizes.txt)
echo "Nr = $NR, PETSC_OPTIONS = '$PETSC_OPTIONS'"
if [ -n "$NR" ] && (( NR % NTASKS != 0 && NTASKS % NR != 0 )); then
    echo "warning: NTASKS = $NTASKS is neither a divisor nor a multiple of Nr = $NR, the radial blocks are uneven or skipped"
fi

CSV=$OUTDIR/timings_N${NODES}.csv
echo "nprocs,nodes,repeat,exit_code,wall_s,scale,method,its,status,setup_s,solve_s,res,err" > $CSV

for r in $(seq 1 $REPEATS); do
    echo "=== $NODES nodes, $NTASKS ranks ($RANKS_PER_NODE per node, $CPUS_PER_TASK cores per rank), run $r ==="
    LOG=$OUTDIR/radius_N${NODES}_n${NTASKS}_r${r}.log

    t0=$(date +%s.%N)
    srun --nodes=$NODES --ntasks=$NTASKS --ntasks-per-node=$RANKS_PER_NODE \
         --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores \
         stdbuf -oL $EXE > $LOG 2>&1
    code=$?
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc -l)

    # "<scale> <label> its N <status> setup X s solve Y s res R err E"
    grep " its .* setup .* solve " $LOG |
        sed -E 's/^(\S+) +(.*\S) +its +([0-9]+) +(\S+) +setup +(\S+) s +solve +(\S+) s +res +(\S+) +err +(.*)$/\1,"\2",\3,\4,\5,\6,\7,\8/' |
        while read -r line; do
            echo "$NTASKS,$NODES,$r,$code,$wall,$line" >> $CSV
        done
    echo "exit code: $code   wall: ${wall} s"
done

echo
cat $CSV
echo "Outputs in $OUTDIR"
