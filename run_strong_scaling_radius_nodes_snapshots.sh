#!/bin/bash
#SBATCH --job-name=radius_nodes_snap
#SBATCH --output=radius_nodes_snap_%j.out
#SBATCH --error=radius_nodes_snap_%j.err
#SBATCH --nodes=1                     # overridden by sbatch --nodes=N
#SBATCH --ntasks-per-node=256         # reserve every core as a task slot
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node
#
# Scaling over the number of NODES of testkinetichypreradius
# (iface/TestKineticHypreRadius.cpp) on the systems dumped from the dataset
# snapshots: every node runs the same FIXED layout, RANKS_PER_NODE MPI ranks x
# THREADS OpenMP threads per rank (default 44 x 4 = 176 cores, the fastest
# layout on one node in run_strong_scaling_radius_cores_snapshots.sh), so the
# total number of ranks grows with the nodes. One job per node count:
#
#   $ for N in 1 2 4 8 16; do sbatch --nodes=$N run_strong_scaling_radius_nodes_snapshots.sh; done
#   $ for N in 1 2 4 8; do SYSTEMS="sys_CQ_tf_L4" sbatch --nodes=$N run_strong_scaling_radius_nodes_snapshots.sh; done
#   $ for N in 1 2 4; do RANKS_PER_NODE=88 THREADS=2 sbatch --nodes=$N run_strong_scaling_radius_nodes_snapshots.sh; done
#
# Every (system, run) solves with RANKS_PER_NODE*N ranks; every solver line of
# the test ("its ... setup ... solve ...") goes to
# $OUTDIR/timings_N<nodes>.csv, one row per (system, run, solver), with the
# same columns as run_strong_scaling_radius_cores_snapshots.sh. All node
# counts of one sweep write to the same $OUTDIR; merge them with
#
#   $ cd $OUTDIR && head -1 timings_N1.csv > timings.csv && tail -q -n +2 timings_N*.csv >> timings.csv
#
# Only hypre/BoomerAMG uses the threads (MKL is kept at 1 thread), so
# THREADS > 1 needs an executable built against the OpenMP PETSc
# (PETSC_MODULE). Every system must already be dumped by dreami (save_sys on)
# in its own folder $DATADIR/<system>, see
# run_strong_scaling_radius_cores_snapshots.sh.
#
# The radial block Jacobi needs the total number of ranks to be a divisor of
# Nr or a multiple of it: with RANKS_PER_NODE a multiple of Nr every node
# count works (44 = 4 x 11, each radius split over 4*N ranks).

PETSC_MODULE=${PETSC_MODULE:-petsc/3.25.2-intelMPI-intel-hypre_omp}

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

CORES_PER_NODE=256                    # 2 sockets x 128 cores, 1 thread per core
RANKS_PER_NODE=${RANKS_PER_NODE:-44}
THREADS=${THREADS:-4}
REPEATS=${REPEATS:-2}
SYSTEMS=${SYSTEMS:-"sys_CQ_tf_L4 sys_injection_t0_L4 sys_injection_tf_L4"}

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close            # threads on the cores next to their rank
export OMP_PLACES=cores

export PETSC_OPTIONS=${PETSC_OPTIONS:-"-scale diag"}

# diagnostics: Intel MPI prints the fabric/provider in use and the rank pinning
export I_MPI_DEBUG=5

NODES=$SLURM_JOB_NUM_NODES
NTASKS=$((RANKS_PER_NODE * NODES))
if (( RANKS_PER_NODE * THREADS > CORES_PER_NODE )); then
    echo "error: $RANKS_PER_NODE ranks x $THREADS threads > $CORES_PER_NODE cores of a node" >&2
    exit 1
fi
# spread the ranks over all cores of the node (both sockets); the THREADS
# threads of a rank take the first cores of its share
CPUS_PER_TASK=$((CORES_PER_NODE / RANKS_PER_NODE))

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
EXE=$DREAM_ROOT/build/iface/testkinetichypreradius
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=$DATADIR/scaling_radius_nodes_snapshots_ppn${RANKS_PER_NODE}_t${THREADS}   # all node counts of one sweep
mkdir -p $OUTDIR

# system to solve in each folder, passed to the test as -mat / -rhs / -ref_solution
MATFILE=${MATFILE:-petsc_mat_np1_step1_iter1}
RHSFILE=${RHSFILE:-petsc_rhs_np1_step1_iter1}
REFFILE=${REFFILE:-petsc_solution_np1_step1_iter1}
export PETSC_OPTIONS="$PETSC_OPTIONS -mat $MATFILE -rhs $RHSFILE -ref_solution $REFFILE"

# check every system first: do not find out about a missing file in the middle of the job
if [ ! -f $EXE ]; then
    echo "'$EXE' not found." >&2
    exit 1
fi
for sys in $SYSTEMS; do
    for f in $MATFILE $RHSFILE $REFFILE petsc_block_sizes.txt; do
        if [ ! -f $DATADIR/$sys/$f ]; then
            echo "'$f' not found (in $DATADIR/$sys)." >&2
            exit 1
        fi
    done
done

# the PETSc the executable actually loads: must be the OpenMP one for THREADS > 1
echo "PETSc library used by $EXE:"
ldd $EXE | grep -E "libpetsc|libHYPRE"

CSV=$OUTDIR/timings_N${NODES}.csv
echo "system,nprocs,threads,cores,nodes,repeat,exit_code,wall_s,scale,method,its,status,setup_s,solve_s,res,err" > $CSV

for sys in $SYSTEMS; do
    cd $DATADIR/$sys
    mkdir -p $OUTDIR/$sys

    NR=$(awk '$1 == "Nr" {print $2}' petsc_block_sizes.txt)
    echo "##### $sys: Nr = $NR, $NODES nodes, PETSC_OPTIONS = '$PETSC_OPTIONS' #####"
    if [ -n "$NR" ] && (( NTASKS > NR && NTASKS % NR != 0 )); then
        echo "skipping $sys: $NTASKS ranks > Nr = $NR and not a multiple of it, the test would skip the radial block Jacobi"
        continue
    fi

    for r in $(seq 1 $REPEATS); do
        echo "=== $sys: $NODES nodes x $RANKS_PER_NODE ranks x $THREADS threads ($NTASKS ranks, $((NTASKS * THREADS)) cores, $CPUS_PER_TASK cores per rank), run $r ==="
        LOG=$OUTDIR/$sys/radius_N${NODES}_n${NTASKS}_t${THREADS}_r${r}.log

        t0=$(date +%s.%N)
        # explicit per-step layout: overrides the job's ntasks-per-node/cpus-per-task inherited from the environment
        srun --nodes=$NODES --ntasks=$NTASKS --ntasks-per-node=$RANKS_PER_NODE \
             --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores \
             stdbuf -oL $EXE > $LOG 2>&1
        code=$?
        t1=$(date +%s.%N)
        wall=$(echo "$t1 - $t0" | bc -l)

        # "<scale> <label> its N <status> setup X s solve Y s res R err E"
        lines=$(grep " its .* setup .* solve " $LOG |
            sed -E 's/^(\S+) +(.*\S) +its +([0-9]+) +(\S+) +setup +(\S+) s +solve +(\S+) s +res +(\S+) +err +(.*)$/\1,"\2",\3,\4,\5,\6,\7,\8/')
        # no solver line: still record the run, so skips and crashes show up in the CSV
        if [ -z "$lines" ]; then
            if grep -q "Skipping radial block Jacobi" $LOG; then st=SKIPPED; else st=FAILED; fi
            lines=",,,$st,,,,"
        fi
        while read -r line; do
            echo "$sys,$NTASKS,$THREADS,$((NTASKS * THREADS)),$NODES,$r,$code,$wall,$line" >> $CSV
        done <<< "$lines"
        echo "exit code: $code   wall: ${wall} s"
    done
done

echo
cat $CSV
echo "Outputs in $OUTDIR"
