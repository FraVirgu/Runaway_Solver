#!/bin/bash
#SBATCH --job-name=conv_tor
#SBATCH --output=conv_tor_%j.out
#SBATCH --error=conv_tor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --account=FUPA2_REDISMHD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --mem=200G
#
# Convergence of the toroidal kinetic runs (examples/kinetic_snapshot_toroidal.py)
# at fixed momentum resolution (LEVEL), all in serial (1 rank):
#
#   radial scan  Nr in NR_LIST, one time step over the snapshot dt
#   time scan    Nr = NR_TIME, the snapshot dt split in NT_LIST steps
#
# For every snapshot in SNAPSHOTS it writes the settings (unless already
# present), runs dreami in OUTDIR/<snapshot>/<case>/ and finally calls
# convergence_toroidal.py, which compares every run with the finest one of
# its scan (moments of f_re and of the f_hot tail, see that script).
#
# The physics options of the settings script (ADV, DBB, AR_FACTOR, ...) pass
# through the environment unchanged.
#
#   $ sbatch run_convergence_toroidal.sh                                   # CQ_t20ms, L1
#   $ SNAPSHOTS="CQ_t5ms CQ_t20ms CQ_t40ms" sbatch run_convergence_toroidal.sh
#   $ NR_LIST="11 22 44" NT_LIST="1 2 4" sbatch run_convergence_toroidal.sh
#
# Memory: serial runs hold the whole grid; at L1 Nr = 44 needs ~7 GB,
# Nr = 88 ~14 GB.

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
module load python                    # settings script and analysis (h5py, matplotlib)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export PETSC_OPTIONS=${PETSC_OPTIONS:-"-dream_kinetic_only -dream_scale diag -dream_rbj_ksp_converged_reason"}

export LEVEL=${LEVEL:-1}
SNAPSHOTS=${SNAPSHOTS:-"CQ_t20ms"}
NR_LIST=${NR_LIST:-"11 22 44 88"}
NR_TIME=${NR_TIME:-44}
NT_LIST=${NT_LIST:-"1 2 4 8"}

DREAM_ROOT=${DREAM_ROOT:-$HOME/DREAM}
DATADIR=${DATADIR:-$DREAM_ROOT/build/iface}
HERE=${SLURM_SUBMIT_DIR:-$(dirname $(readlink -f $0))}
OUTDIR=${OUTDIR:-$DATADIR/convergence_toroidal_L${LEVEL}_${SLURM_JOB_ID:-local}}
DREAMI=$DREAM_ROOT/build/iface/dreami

if [ ! -x $DREAMI ]; then
    echo "'$DREAMI' not found: build it first." >&2
    exit 1
fi

mkdir -p $OUTDIR
echo "LEVEL = $LEVEL, SNAPSHOTS = '$SNAPSHOTS'"
echo "radial scan Nr = '$NR_LIST' (1 step); time scan Nr = $NR_TIME, steps = '$NT_LIST'"
echo "PETSC_OPTIONS = '$PETSC_OPTIONS'"
echo "Outputs in $OUTDIR"

# run_case <snapshot> <scan> <nr> <nt>
run_case() {
    local snap=$1 scan=$2 nr=$3 nt=$4
    local tag="${snap}_nr${nr}$([ $nt -gt 1 ] && echo _nt$nt)"
    local settings=$DATADIR/dream_settings_toroidal_${tag}_L${LEVEL}.h5
    if [ ! -f $settings ]; then
        if [[ $snap == CQ* ]]; then ntvar=NT_CQ; else ntvar=NT_INJ; fi
        env SNAPSHOT=$snap NR=$nr $ntvar=$nt KINETIC_OUT_DIR=$DATADIR \
            python3 $HERE/examples/kinetic_snapshot_toroidal.py > $OUTDIR/settings_${tag}.log 2>&1
    fi
    if [ ! -f $settings ]; then
        echo "=== $snap $scan nr$nr nt$nt: settings not written, see $OUTDIR/settings_${tag}.log"
        return
    fi
    local dir=$OUTDIR/$snap/${scan}_nr${nr}_nt${nt}
    mkdir -p $dir && cd $dir
    local t0=$(date +%s.%N)
    srun --ntasks=1 --cpus-per-task=1 $DREAMI $settings > dreami.log 2>&1
    local wall=$(echo "$(date +%s.%N) - $t0" | bc -l)
    echo $wall > wall_s.txt
    if grep -q "Saved output" dreami.log; then
        echo "=== $snap $scan nr$nr nt$nt: OK in $wall s  ($(grep -h 'dream_rbj_ solve' dreami.log | tail -1 | sed 's/.*due to //'))"
    else
        echo "=== $snap $scan nr$nr nt$nt: FAILED after $wall s, see $dir/dreami.log"
    fi
}

for snap in $SNAPSHOTS; do
    for nr in $NR_LIST; do run_case $snap radial $nr 1; done
    for nt in $NT_LIST; do run_case $snap time $NR_TIME $nt; done
done

echo
python3 $HERE/convergence_toroidal.py $OUTDIR --plot $OUTDIR/convergence.png
