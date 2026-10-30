#!/bin/bash
#SBATCH --job-name=radius_cores_snap
#SBATCH --output=radius_cores_snap_%j.out
#SBATCH --error=radius_cores_snap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256         # reserve every core as a task slot, so steps up to 256 ranks fit
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00             # the 4 h of run_strong_scaling_radius_cores.sh for each of the 3 systems
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node
#
# Strong scaling of testkinetichypreradius (iface/TestKineticHypreRadius.cpp)
# on the systems dumped from the dataset snapshots, as
# run_strong_scaling_radius_cores.sh does for one system: on ONE node, hybrid
# MPI + OpenMP, every system in SYSTEMS is solved with n MPI ranks x t OpenMP
# threads per rank, for every n in NPROCS and t in THREADS with n*t <= 256
# cores. Every (system, n, t) is run REPEATS times; every solver line of the
# test ("its ... setup ... solve ...") is collected in $OUTDIR/timings.csv, one
# row per (system, n, t, run, solver), and every run keeps its own log in
# $OUTDIR/<system>/.
#
# Only hypre/BoomerAMG uses the threads (MKL is kept at 1 thread), so t > 1
# needs an executable built against the OpenMP PETSc (PETSC_MODULE).
#
# Every system must already be dumped by dreami (save_sys on) in its own folder
# $DATADIR/<system>:
#   petsc_mat_np1_step1_iter1, petsc_rhs_np1_step1_iter1,
#   petsc_block_sizes.txt and the reference solution
#   petsc_solution_np1_step1_iter1. The test moves the fluid unknowns to the
#   right-hand side with it: without it they are taken as 0.
#
# Submit with
#
#   $ sbatch run_strong_scaling_radius_cores_snapshots.sh
#   $ SYSTEMS="sys_CQ_tf_L4" sbatch run_strong_scaling_radius_cores_snapshots.sh
#   $ NPROCS="16 32 64" THREADS="1 4 16" REPEATS=3 sbatch run_strong_scaling_radius_cores_snapshots.sh
#   $ MATFILE=my_mat RHSFILE=my_rhs REFFILE=my_sol sbatch run_strong_scaling_radius_cores_snapshots.sh
#
# The radial block Jacobi needs the number of ranks n to be a divisor of Nr
# (whole radii per rank) or a multiple of it (each radius split over n/Nr
# ranks); other n are skipped by the test.

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

# OMP_NUM_THREADS is set per run below
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close            # threads on the cores next to their rank
export OMP_PLACES=cores

export PETSC_OPTIONS=${PETSC_OPTIONS:-"-scale diag"}

# diagnostics: Intel MPI prints the fabric/provider in use and the rank pinning
export I_MPI_DEBUG=5

CORES_PER_NODE=256                    # 2 sockets x 128 cores, 1 thread per core
NPROCS=${NPROCS:-"1 2 4 8 16 32 64 128 256"}
THREADS=${THREADS:-"1 2 4 8 16"}
REPEATS=${REPEATS:-2}
SYSTEMS=${SYSTEMS:-"sys_CQ_tf_L4 sys_injection_t0_L4 sys_injection_tf_L4"}

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
EXE=$DREAM_ROOT/build/iface/testkinetichypreradius
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=$DATADIR/scaling_radius_cores_snapshots_${SLURM_JOB_ID}
mkdir -p $OUTDIR

# system to solve in each folder, passed to the test as -mat / -rhs / -ref_solution
MATFILE=${MATFILE:-petsc_mat_np1_step1_iter1}
RHSFILE=${RHSFILE:-petsc_rhs_np1_step1_iter1}
REFFILE=${REFFILE:-petsc_solution_np1_step1_iter1}
export PETSC_OPTIONS="$PETSC_OPTIONS -mat $MATFILE -rhs $RHSFILE -ref_solution $REFFILE"

# check every system first: do not find out about a missing file hours into the job
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

# the PETSc the executable actually loads: must be the OpenMP one for t > 1
echo "PETSc library used by $EXE:"
ldd $EXE | grep -E "libpetsc|libHYPRE"

CSV=$OUTDIR/timings.csv
echo "system,nprocs,threads,cores,nodes,repeat,exit_code,wall_s,scale,method,its,status,setup_s,solve_s,res,err" > $CSV

for sys in $SYSTEMS; do
    cd $DATADIR/$sys
    mkdir -p $OUTDIR/$sys

    NR=$(awk '$1 == "Nr" {print $2}' petsc_block_sizes.txt)
    echo "##### $sys: Nr = $NR, 1 node, PETSC_OPTIONS = '$PETSC_OPTIONS' #####"

    for n in $NPROCS; do
        if (( n > CORES_PER_NODE || CORES_PER_NODE % n != 0 )); then
            echo "skipping n = $n: must divide the $CORES_PER_NODE cores of the node" >&2
            continue
        fi
        if [ -n "$NR" ] && (( NR % n != 0 && n % NR != 0 )); then
            echo "warning: n = $n is neither a divisor nor a multiple of Nr = $NR, the radial block Jacobi is skipped"
        fi
        # spread the ranks over all cores of the node (both sockets); the t
        # threads of a rank take the first t cores of its share
        CPUS_PER_TASK=$((CORES_PER_NODE / n))

        for t in $THREADS; do
            if (( n * t > CORES_PER_NODE )); then
                echo "skipping n = $n, t = $t: $((n * t)) cores > $CORES_PER_NODE cores of the node"
                continue
            fi
            export OMP_NUM_THREADS=$t

            for r in $(seq 1 $REPEATS); do
                echo "=== $sys: $n ranks x $t threads ($((n * t)) cores, $CPUS_PER_TASK cores per rank), run $r ==="
                LOG=$OUTDIR/$sys/radius_n${n}_t${t}_r${r}.log

                t0=$(date +%s.%N)
                # explicit per-step layout: overrides the job's ntasks-per-node/cpus-per-task inherited from the environment
                srun --nodes=1 --ntasks=$n --ntasks-per-node=$n --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores \
                     stdbuf -oL $EXE > $LOG 2>&1
                code=$?
                t1=$(date +%s.%N)
                wall=$(echo "$t1 - $t0" | bc -l)

                # "<scale> <label> its N <status> setup X s solve Y s res R err E"
                grep " its .* setup .* solve " $LOG |
                    sed -E 's/^(\S+) +(.*\S) +its +([0-9]+) +(\S+) +setup +(\S+) s +solve +(\S+) s +res +(\S+) +err +(.*)$/\1,"\2",\3,\4,\5,\6,\7,\8/' |
                    while read -r line; do
                        echo "$sys,$n,$t,$((n * t)),1,$r,$code,$wall,$line" >> $CSV
                    done
                echo "exit code: $code   wall: ${wall} s"
            done
        done
    done
done

echo
cat $CSV
echo "Outputs in $OUTDIR"
