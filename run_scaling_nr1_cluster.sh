#!/bin/bash
#SBATCH --job-name=scal_nr1
#SBATCH --output=scal_nr1_%j.out
#SBATCH --error=scal_nr1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64          # up to 64 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=0-04:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --exclusive                   # clean timings, no other jobs on the node
#
# Strong scaling, ONE radial cell (Nr = 1): the single radius block is split
# over all n MPI ranks (n = 1 ... 64) and solved by testkinetichypreradius
# (iface/TestKineticHypreRadius.cpp), which runs two variants:
#   a. GMRES + one BoomerAMG V-cycle                    ("AMG + GMRES")
#   b. FGMRES + [inner GMRES (rtol 1e-6) + BoomerAMG]   (MIRadialBJ)
# Every (system, n) is run with two BoomerAMG smoothers:
#   default   hypre's default (hybrid Gauss-Seidel)
#   euclid    Euclid ILU(1)                             ("AMG/Euclid + GMRES")
# so each run gives 4 lines: {AMG, AMG/Euclid} x {GMRES, [GMRES]+FGMRES}.
#
# The systems are the ones dumped by run_compare_solvers_nr1_cluster.sh:
#   $DATADIR/sys_toroidal_<snapshot>_nr1_L<LEVEL>/
#     petsc_mat_np1_step1_iter1, petsc_rhs_np1_step1_iter1,
#     petsc_solution_np1_step1_iter1, petsc_block_sizes.txt
#
# Results: $OUTDIR/timings.csv, one row per (system, n, smoother, run,
# solver), and one log per run in $OUTDIR/<system>/.
#
# Submit from the Runaway_Solver folder with
#
#   $ sbatch run_scaling_nr1_cluster.sh                          # L2, all snapshots
#   $ SNAPSHOTS="CQ_t49ms CQ_tf" sbatch run_scaling_nr1_cluster.sh
#   $ LEVEL=1 NPROCS="1 2 4 8" REPEATS=3 sbatch run_scaling_nr1_cluster.sh

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

LEVEL=${LEVEL:-2}
NPROCS=${NPROCS:-"1 2 4 8 16 32 64"}
REPEATS=${REPEATS:-2}
SMOOTHERS=${SMOOTHERS:-"default euclid"}

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
EXE=$DREAM_ROOT/build/iface/testkinetichypreradius
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
OUTDIR=$DATADIR/scaling_nr1_L${LEVEL}_${SLURM_JOB_ID:-local}
mkdir -p $OUTDIR

MATFILE=petsc_mat_np1_step1_iter1
RHSFILE=petsc_rhs_np1_step1_iter1
REFFILE=petsc_solution_np1_step1_iter1
BASE_OPTIONS="-scale diag -mat $MATFILE -rhs $RHSFILE -ref_solution $REFFILE"
EUCLID_OPTIONS="-radbj_sub_pc_hypre_boomeramg_smooth_type Euclid -radbj_sub_pc_hypre_boomeramg_eu_level 1 \
-radbjin_sub_pc_hypre_boomeramg_smooth_type Euclid -radbjin_sub_pc_hypre_boomeramg_eu_level 1"

if [ ! -x $EXE ]; then
    echo "'$EXE' not found: build it first (make testkinetichypreradius)." >&2
    exit 1
fi

# all dumped Nr = 1 systems at this LEVEL, unless SNAPSHOTS is given
if [ -z "$SNAPSHOTS" ]; then
    for d in $DATADIR/sys_toroidal_*_nr1_L${LEVEL}; do
        [ -d "$d" ] || continue
        s=$(basename $d)
        s=${s#sys_toroidal_}
        SNAPSHOTS="$SNAPSHOTS ${s%_nr1_L${LEVEL}}"
    done
fi
SYSTEMS=""
for snap in $SNAPSHOTS; do
    sys=sys_toroidal_${snap}_nr1_L${LEVEL}
    ok=1
    for f in $MATFILE $RHSFILE $REFFILE petsc_block_sizes.txt; do
        [ -f $DATADIR/$sys/$f ] || { echo "'$f' not found in $DATADIR/$sys: skipped" >&2; ok=0; break; }
    done
    (( ok )) && SYSTEMS="$SYSTEMS $sys"
done
if [ -z "$SYSTEMS" ]; then
    echo "No dumped Nr = 1 systems at LEVEL $LEVEL in $DATADIR: run run_compare_solvers_nr1_cluster.sh first." >&2
    exit 1
fi
echo "Systems:$SYSTEMS"
echo "NPROCS = $NPROCS, smoothers = $SMOOTHERS, $REPEATS repeats"

CSV=$OUTDIR/timings.csv
echo "system,nprocs,smoother,repeat,exit_code,wall_s,scale,method,its,status,setup_s,solve_s,res,err" > $CSV

for sys in $SYSTEMS; do
    cd $DATADIR/$sys
    mkdir -p $OUTDIR/$sys
    echo "##### $sys #####"

    for n in $NPROCS; do
        for sm in $SMOOTHERS; do
            if [ "$sm" = euclid ]; then
                export PETSC_OPTIONS="$BASE_OPTIONS $EUCLID_OPTIONS"
            else
                export PETSC_OPTIONS="$BASE_OPTIONS"
            fi

            for r in $(seq 1 $REPEATS); do
                LOG=$OUTDIR/$sys/n${n}_${sm}_r${r}.log
                t0=$(date +%s.%N)
                srun --nodes=1 --ntasks=$n --cpus-per-task=1 --cpu-bind=cores \
                     stdbuf -oL $EXE > $LOG 2>&1
                code=$?
                t1=$(date +%s.%N)
                wall=$(echo "$t1 - $t0" | bc -l)

                # "<scale> <label> its N <status> setup X s solve Y s res R err E"
                lines=$(grep " its .* setup .* solve " $LOG | grep -v "reference" |
                    sed -E 's/^(\S+) +(.*\S) +its +([0-9]+) +(\S+) +setup +(\S+) s +solve +(\S+) s +res +(\S+) +err +(.*)$/\1,"\2",\3,\4,\5,\6,\7,\8/')
                [ -z "$lines" ] && lines=",,,FAILED,,,,"
                while read -r line; do
                    echo "$sys,$n,$sm,$r,$code,$wall,$line" >> $CSV
                done <<< "$lines"
                echo "$sys  n = $n  $sm  run $r: exit $code, wall ${wall} s"
                grep -E " its .* setup .* solve |inner GMRES iterations" $LOG | grep -v "reference" | sed 's/^/    /'
            done
        done
    done
done

echo
echo "CSV: $CSV"
