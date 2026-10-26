#!/bin/bash
#SBATCH --job-name=radius_scaling
#SBATCH --output=radius_scaling_%j.out
#SBATCH --error=radius_scaling_%j.err
#SBATCH --nodes=1                     # enough nodes for the largest n in NPROCS
#SBATCH --time=0-04:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node
#
# Strong scaling of testkinetichypreradius (iface/TestKineticHypreRadius.cpp):
# the same kinetic system with Nr radial cells is solved with n = NPROCS MPI
# ranks. Every n is run REPEATS times; every solver line of the test
# ("its ... setup ... solve ...") is collected in $OUTDIR/timings.csv, one
# row per (n, run, solver), and every run keeps its own log.
#
# The system must already be dumped in $DATADIR by dreami (save_sys on):
#   petsc_mat_np1_step1_iter1, petsc_rhs_np1_step1_iter1,
#   petsc_block_sizes.txt and, optionally, the reference solution.
# e.g.
#   $ LEVEL=2 NR=64 python examples/diff_grid_paper_kinetic_radial.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_paper_radial_L2_Nr64.h5
#
# Submit with
#
#   $ sbatch run_strong_scaling_radius.sh
#   $ NPROCS="1 2 4 8 16 32 64 128 256 512" sbatch --nodes=2 run_strong_scaling_radius.sh
#
# The radial block-Jacobi/ASM part needs whole radii on every rank, so it is
# skipped by the test when n > Nr; choose Nr as a multiple of the largest n.

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

CORES_PER_NODE=256
NPROCS=${NPROCS:-"1 2 4 8 16 32 64 128 256"}
REPEATS=${REPEATS:-2}

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
EXE=$DREAM_ROOT/build/iface/testkinetichypreradius
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=$DATADIR/scaling_radius_${SLURM_JOB_ID}
mkdir -p $OUTDIR
cd $DATADIR

for f in $EXE petsc_mat_np1_step1_iter1 petsc_rhs_np1_step1_iter1 petsc_block_sizes.txt; do
    if [ ! -f $f ]; then
        echo "'$f' not found (in $DATADIR)." >&2
        exit 1
    fi
done

NR=$(awk '$1 == "Nr" {print $2}' petsc_block_sizes.txt)
NODES=$SLURM_JOB_NUM_NODES
echo "Nr = $NR, $NODES node(s), PETSC_OPTIONS = '$PETSC_OPTIONS'"

CSV=$OUTDIR/timings.csv
echo "nprocs,nodes,repeat,exit_code,wall_s,scale,method,its,status,setup_s,solve_s,res,err" > $CSV

for n in $NPROCS; do
    # fill the nodes one after the other: as few nodes as possible for n ranks
    nodes=$(( (n + CORES_PER_NODE - 1) / CORES_PER_NODE ))
    if (( nodes > NODES )); then
        echo "skipping n = $n: needs $nodes nodes, the job has $NODES" >&2
        continue
    fi
    if [ -n "$NR" ] && (( n > NR || NR % n != 0 )); then
        echo "warning: Nr = $NR is not a multiple of n = $n, the radial blocks are uneven or skipped"
    fi

    for r in $(seq 1 $REPEATS); do
        echo "=== $n ranks on $nodes node(s), run $r ==="
        LOG=$OUTDIR/radius_n${n}_r${r}.log

        t0=$(date +%s.%N)
        srun --nodes=$nodes --ntasks=$n --cpu-bind=cores \
             stdbuf -oL $EXE > $LOG 2>&1
        code=$?
        t1=$(date +%s.%N)
        wall=$(echo "$t1 - $t0" | bc -l)

        # "<scale> <label> its N <status> setup X s solve Y s res R err E"
        grep " its .* setup .* solve " $LOG |
            sed -E 's/^(\S+) +(.*\S) +its +([0-9]+) +(\S+) +setup +(\S+) s +solve +(\S+) s +res +(\S+) +err +(.*)$/\1,"\2",\3,\4,\5,\6,\7,\8/' |
            while read -r line; do
                echo "$n,$nodes,$r,$code,$wall,$line" >> $CSV
            done
        echo "exit code: $code   wall: ${wall} s"
    done
done

echo
cat $CSV
echo "Outputs in $OUTDIR"
