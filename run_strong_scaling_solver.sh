#!/bin/bash
#SBATCH --job-name=kin_scaling
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --nodes=2                     # overridden by sbatch --nodes=N
#SBATCH --time=0-02:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node

# Fixed total number of MPI ranks, spread over the nodes of this job.
# One job per node count:
#   for N in 2 4 8 16 32 64; do NTASKS=128 sbatch --nodes=$N run_strong_scaling_solver.sh; done
# Add RUN_SERIAL=1 to one of the submissions to also run the 1-rank serial version.

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

# $SLURM_CPU_PER_TASK
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

export PETSC_OPTIONS="-scale diag"

# diagnostics: Intel MPI prints the fabric/provider in use and the rank pinning
export I_MPI_DEBUG=5

CORES_PER_NODE=256
NTASKS=${NTASKS:-128}                 # total MPI ranks, the same for every node count
RUN_SERIAL=${RUN_SERIAL:-0}

NODES=$SLURM_JOB_NUM_NODES
if (( NTASKS % NODES != 0 )); then
    echo "error: NTASKS=$NTASKS is not divisible by $NODES nodes" >&2
    exit 1
fi
RANKS_PER_NODE=$((NTASKS / NODES))
if (( RANKS_PER_NODE > CORES_PER_NODE )); then
    echo "error: $RANKS_PER_NODE ranks per node > $CORES_PER_NODE cores" >&2
    exit 1
fi
# spread the ranks over all cores of the node (both sockets)
CPUS_PER_TASK=$((CORES_PER_NODE / RANKS_PER_NODE))

TESTDIR=$HOME/DREAM/build/iface
OUTDIR=$TESTDIR/scaling_n${NTASKS}    # all node counts of one sweep in the same directory
mkdir -p $OUTDIR
cd $TESTDIR

# ---- serial version, 1 MPI rank ----
if (( RUN_SERIAL )); then
    echo "=== serial, 1 rank ==="
    srun --nodes=1 --ntasks=1 --cpus-per-task=2 --cpu-bind=cores \
         stdbuf -oL ./testkinetichypreserial > $OUTDIR/serial_n1.log 2>&1
    echo "exit code: $?"
fi

# ---- parallel version, NTASKS MPI ranks on NODES nodes ----
echo "=== parallel, $NODES nodes, $NTASKS ranks ($RANKS_PER_NODE per node, $CPUS_PER_TASK cores per rank) ==="
srun --nodes=$NODES --ntasks=$NTASKS --ntasks-per-node=$RANKS_PER_NODE \
     --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores \
     stdbuf -oL ./testkinetichypreparallel > $OUTDIR/parallel_N${NODES}_n${NTASKS}.log 2>&1
echo "exit code: $?"

echo "Outputs in $OUTDIR"
