#!/bin/bash
#SBATCH --job-name=dreami_scaling
#SBATCH --output=dreami_scaling_%j.out
#SBATCH --error=dreami_scaling_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16         # the largest run
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node
#
# Strong scaling of dreami in kinetic-only mode, n = 1 .. 32 MPI ranks.
# Every n is run REPEATS times; the timings are collected in
# $OUTDIR/timings.csv (one row per run) and every run keeps its own log
# and output file.
#
# Submit with
#
#   $ sbatch run_scaling_dreami.sh
#
# The settings file must already exist in $TESTDIR (python basic_kinetic_only.py).

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

export PETSC_OPTIONS="-dream_kinetic_only -dream_scale diag -dream_amg_lag_pc 20"

NPROCS="1 2 4 8 16 32"
REPEATS=2

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
TESTDIR=$DREAM_ROOT/build/iface
SETTINGS=dream_settings_paper.h5
OUTPUT=output_paper.h5                # output/filename in the settings file
OUTDIR=$TESTDIR/scaling_${SLURM_JOB_ID}
mkdir -p $OUTDIR
cd $TESTDIR

if [ ! -f $SETTINGS ]; then
    echo "Settings file '$TESTDIR/$SETTINGS' not found."
    exit 1
fi

# Convert DREAM's "Solved equation system in 1 minute, 2 seconds, 3 milliseconds,
# 4 microseconds." into seconds.
solve_time() {
    grep "Solved equation system in" $1 | tail -n 1 | sed 's/.*system in //; s/\.$//' |
        awk -F', ' '{
            t = 0
            for (i = 1; i <= NF; i++) {
                split($i, a, " ")
                if      (a[2] ~ /^hour/)        t += a[1] * 3600
                else if (a[2] ~ /^minute/)      t += a[1] * 60
                else if (a[2] ~ /^second/)      t += a[1]
                else if (a[2] ~ /^millisecond/) t += a[1] * 1e-3
                else if (a[2] ~ /^microsecond/) t += a[1] * 1e-6
            }
            printf "%.6f", t
        }'
}

CSV=$OUTDIR/timings.csv
echo "nprocs,repeat,exit_code,wall_s,solve_s,invert_s" > $CSV

for n in $NPROCS; do
    for r in $(seq 1 $REPEATS); do
        echo "=== $n ranks, run $r ==="
        LOG=$OUTDIR/dreami_n${n}_r${r}.log
        rm -f $OUTPUT

        t0=$(date +%s.%N)
        srun --ntasks=$n --cpu-bind=cores ./dreami $SETTINGS > $LOG 2>&1
        code=$?
        t1=$(date +%s.%N)

        wall=$(echo "$t1 - $t0" | bc -l)
        solve=$(solve_time $LOG)
        invert=$(grep "Total invert time" $LOG | tail -n 1 | awk '{print $(NF-1)}')
        echo "$n,$r,$code,$wall,$solve,$invert" >> $CSV
        echo "exit code: $code   wall: ${wall} s   solve: ${solve} s   invert: ${invert} s"

        [ -f $OUTPUT ] && mv $OUTPUT $OUTDIR/output_n${n}_r${r}.h5
    done
done

echo
column -s, -t $CSV
echo "Outputs in $OUTDIR"
