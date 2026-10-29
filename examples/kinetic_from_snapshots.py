#!/usr/bin/env python3
#
# Kinetic-only DREAM runs started from the snapshots in DATASET/snapshots/
# (written by DATASET/extract_snapshots.py), one independent run per
# snapshot.
#
# Each run evolves f_hot and f_re ONE time step, with the background held
# fixed at the snapshot values:
#
#   E_field(r), T_cold(r), n_cold(r), n_i(Z0, r)    prescribed, from state.h5
#   dt                                              time step of the source
#                                                   simulation at the snapshot
#
# The numerics are those of diff_grid_paper_kinetic_radial.py: two kinetic
# grids of different resolution, Rechester-Rosenbluth transport plus radial
# advection on both, synchrotron damping, no knock-on source, and the
# LINEAR_IMPLICIT solver.
#
# Initial condition:
#   f_hot   Maxwellian with the snapshot n_cold(r), T_cold(r)
#   f_re    beam exp(-(p-P_RE)/RE_PT) exp(-(1-xi)/RE_XIW), normalised to the
#           snapshot n_re(r) (the fluid run only stores the density); zero
#           if n_re = 0
#
# Geometry: cylindrical with B0 = B_min on axis and the snapshot radial grid.
# The source runs use the numerical ITER equilibrium, which is not available.
#
# Run as
#
#   $ ./kinetic_from_snapshots.py                       # all snapshots
#   $ SNAPSHOT=CQ_tf LEVEL=1 ./kinetic_from_snapshots.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_kinetic_CQ_tf_L2.h5
#
# ###################################################################

import glob
import os
import sys
import h5py
import numpy as np

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', os.path.join(EXAMPLES_DIR, '..', '..', 'DREAM')))
OUT_DIR = os.path.expanduser(
    os.environ.get('KINETIC_OUT_DIR', os.path.join(DREAM_ROOT, 'build', 'iface')))
DATASET_DIR = os.path.normpath(os.path.join(EXAMPLES_DIR, '..', 'DATASET'))
SNAP_DIR = os.path.join(DATASET_DIR, 'snapshots')

sys.path.append(os.path.join(DREAM_ROOT, 'py'))
sys.path.append(DATASET_DIR)

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport

from extract_snapshots import SNAPSHOTS

mc2 = 510998.95             # electron rest energy [eV]

# =====================================================================
# Numerical parameters (as in diff_grid_paper_kinetic_radial.py)
# =====================================================================
P_RE  = 1.0                 # hot/runaway grid interface
P_MAX = 60.0

LEVEL = int(os.environ.get('LEVEL', 4))
_f = 2 ** LEVEL

NP_HOT  = 125 * _f          # many p
NXI_HOT =  16 * _f          # few xi

NP_RE   =  75 * _f          # few p
NXI_RE  = 100 * _f          # many xi

BIUNIFORM_RE   = True
THETASEP       = 0.5
NTHETASEP_FRAC = 0.75

ADV_INTERP = {
    'upwind': DistFunc.AD_INTERP_UPWIND,
    'quick':  DistFunc.AD_INTERP_QUICK,
}[os.environ.get('ADV', 'upwind').lower()]

# Rechester-Rosenbluth dB/B; DBB=0 switches transport off.
DBB = float(os.environ.get('DBB', 1e-4))

# Radial advection A_r(r) = AR_FACTOR * E(r) [m/s]; AR_FACTOR=0 switches it off.
AR_FACTOR = float(os.environ.get('AR_FACTOR', 10.0))

# Shape of the initial runaway beam: e-folding momentum and pitch width.
RE_PT  = float(os.environ.get('RE_PT', 10.0))
RE_XIW = float(os.environ.get('RE_XIW', 0.05))

p_re_init  = np.linspace(P_RE, P_MAX, 1001)
xi_re_init = np.linspace(-1.0, 1.0,    801)


def addRadialAdvection(tr, ar, r):
    """
    Prescribe the radial advection ar(r) [m/s] (constant in t, p, xi) on the
    kinetic transport settings tr, keeping a Rechester-Rosenbluth dB/B set
    before. See diff_grid_paper_kinetic_radial.py: TransportSettings writes
    ar only for type PRESCRIBED, so with both ar is added to the dictionary.
    """
    rr = (tr.type == Transport.TRANSPORT_RECHESTER_ROSENBLUTH)
    tr.prescribeAdvection(np.reshape(ar, (1, ar.size, 1, 1)),
                          t=[0], r=r, p=[0], xi=[0])
    tr.setBoundaryCondition(Transport.BC_F_0)
    if not rr:
        return

    tr.type = Transport.TRANSPORT_RECHESTER_ROSENBLUTH
    todict = tr.todict
    def todict_with_ar():
        data = todict()
        data['ar'] = {'x': tr.ar, 'r': tr.ar_r, 't': tr.ar_t,
                      'interp3d': tr.ar_interp3d, 'p': tr.ar_p, 'xi': tr.ar_xi}
        return data
    tr.todict = todict_with_ar


def snapshot_dt(name):
    """Time step of the source simulation at the snapshot."""
    fname, timeindex = SNAPSHOTS[name]
    with h5py.File(os.path.join(DATASET_DIR, fname), 'r') as f:
        t = f['grid/t'][:]
    k = timeindex % t.size
    return t[1] - t[0] if k == 0 else t[k] - t[k-1]


def load_snapshot(folder):
    with h5py.File(os.path.join(folder, 'state.h5'), 'r') as f:
        s = {q: f['eqsys'][q][0] for q in ('E_field', 'T_cold', 'n_cold', 'n_re', 'n_i')}
        s['r']    = f['grid/r'][:]
        s['r_f']  = f['grid/r_f'][:]
        s['B0']   = f['grid/geometry/Bmin'][0]
        s['wall'] = f['settings/radialgrid/wall_radius'][0]

        ni = f['settings/eqsys/n_i']
        s['ion_names']    = [n for n in ni['names'][()].tobytes().decode().split(';') if n]
        s['ion_Z']        = ni['Z'][:]
        s['ion_isotopes'] = ni['isotopes'][:]
    return s


def f_re_init(n_re, xi0):
    """Runaway beam along xi0 = +-1 with density n_re(r), shape (nr, nxi, np)."""
    P, XI = np.meshgrid(p_re_init, xi_re_init)
    shape = np.exp(-(P - P_RE) / RE_PT) * np.exp(-(1 - xi0 * XI) / RE_XIW)
    # n = 2 pi int int p^2 f dp dxi
    norm = 2 * np.pi * np.trapezoid(np.trapezoid(P**2 * shape, p_re_init, axis=1), xi_re_init)
    return n_re[:, np.newaxis, np.newaxis] * (shape / norm)[np.newaxis, :, :]


def make_settings(name, s, dt):
    r = s['r']
    ds = DREAMSettings()

    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    # -----------------------------------------------------------------
    # Fixed background from the snapshot
    # -----------------------------------------------------------------
    ds.eqsys.E_field.setPrescribedData(s['E_field'], radius=r)
    ds.eqsys.T_cold.setPrescribedData(s['T_cold'], radius=r)
    ds.eqsys.n_cold.setPrescribedData(s['n_cold'], radius=r)

    # n_i is stored as consecutive blocks of Z+1 charge states per species;
    # prescribed ions need a time axis, here a single (constant) time point
    i0 = 0
    for ion, Z, iso in zip(s['ion_names'], s['ion_Z'], s['ion_isotopes']):
        ds.eqsys.n_i.addIon(name=ion, Z=int(Z), isotope=int(iso), iontype=Ions.IONS_PRESCRIBED,
                            n=s['n_i'][i0:i0+Z+1, np.newaxis, :], r=r, t=np.array([0.0]))
        i0 += Z + 1

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

    # -----------------------------------------------------------------
    # Hot-tail grid: [0, P_RE], Maxwellian initial condition
    # -----------------------------------------------------------------
    ds.hottailgrid.setNp(NP_HOT)
    ds.hottailgrid.setNxi(NXI_HOT)
    ds.hottailgrid.setPmax(P_RE)

    ds.eqsys.f_hot.setInitialProfiles(n0=s['n_cold'], T0=s['T_cold'], rn0=r, rT0=r)
    ds.eqsys.f_hot.setBoundaryCondition(DistFunc.BC_F_0)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ADV_INTERP)
    ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)

    # -----------------------------------------------------------------
    # Runaway grid: [P_RE, P_MAX], beam initial condition
    # -----------------------------------------------------------------
    ds.runawaygrid.setEnabled(True)
    ds.runawaygrid.setNp(NP_RE)
    ds.runawaygrid.setNxi(NXI_RE)
    ds.runawaygrid.setPmax(P_MAX)

    if BIUNIFORM_RE:
        ds.runawaygrid.setBiuniformGrid(thetasep=THETASEP,
                                        nthetasep_frac=NTHETASEP_FRAC)

    if np.any(s['n_re'] > 0):
        # Same convention as diff_grid_paper_kinetic_radial.py: E > 0
        # accelerates toward xi = +1
        xi0 = 1.0 if np.mean(s['E_field']) >= 0 else -1.0
        ds.eqsys.f_re.setInitialValue(f_re_init(s['n_re'], xi0), r=r, p=p_re_init, xi=xi_re_init)
    ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
    ds.eqsys.f_re.setAdvectionInterpolationMethod(ADV_INTERP)

    # -----------------------------------------------------------------
    # Geometry: cylindrical on the snapshot radial grid
    # -----------------------------------------------------------------
    ds.radialgrid.setB0(s['B0'])
    ds.radialgrid.setCustomGridPoints(s['r_f'])
    ds.radialgrid.setWallRadius(s['wall'])

    # -----------------------------------------------------------------
    # Radial transport
    # -----------------------------------------------------------------
    if DBB > 0:
        ds.eqsys.f_hot.transport.setMagneticPerturbation(DBB)
        ds.eqsys.f_hot.transport.setBoundaryCondition(Transport.BC_F_0)
        ds.eqsys.f_re.transport.setMagneticPerturbation(DBB)
        ds.eqsys.f_re.transport.setBoundaryCondition(Transport.BC_F_0)

    if AR_FACTOR != 0:
        ar = AR_FACTOR * s['E_field']
        addRadialAdvection(ds.eqsys.f_hot.transport, ar, r)
        addRadialAdvection(ds.eqsys.f_re.transport, ar, r)

    # -----------------------------------------------------------------
    # Solver: one time step of the snapshot dt
    # -----------------------------------------------------------------
    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.solver.preconditioner.setEnabled(False)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_AMG)

    ds.other.include('fluid', 'nu_s', 'nu_D')

    ds.timestep.setTmax(dt)
    ds.timestep.setNt(1)

    ds.output.setTiming(stdout=True, file=True)
    return ds


only = os.environ.get('SNAPSHOT')
folders = sorted(f for f in glob.glob(os.path.join(SNAP_DIR, '*')) if os.path.isdir(f))
if only:
    folders = [f for f in folders if os.path.basename(f) == only]
if not folders:
    sys.exit(f'No snapshots found in {SNAP_DIR}; run DATASET/extract_snapshots.py first.')

os.makedirs(OUT_DIR, exist_ok=True)
N_tot = (NP_HOT * NXI_HOT + NP_RE * NXI_RE) * 11
print('=' * 70)
print(f'Kinetic runs from DATASET snapshots               [LEVEL = {LEVEL}]')
print(f'  hot grid  Np = {NP_HOT}  Nxi = {NXI_HOT}   dp = {P_RE/NP_HOT:.2e}')
print(f'  RE grid   Np = {NP_RE}  Nxi = {NXI_RE}')
print(f'  dB/B = {DBB:.1e}   A_r = {AR_FACTOR:g} E(r)   advection {os.environ.get("ADV", "upwind")}')
print('=' * 70)

for folder in folders:
    name = os.path.basename(folder)
    s = load_snapshot(folder)
    dt = snapshot_dt(name)

    ds = make_settings(name, s, dt)
    TAG = f'kinetic_{name}_L{LEVEL}'
    ds.output.setFilename(f'output_{TAG}.h5')
    settings_file = os.path.join(OUT_DIR, f'dream_settings_{TAG}.h5')
    ds.save(settings_file)

    # The thermal bulk is resolved on the hot grid only if dp << p_th
    p_th = np.sqrt(2 * s['T_cold'].min() / mc2)
    print(f'{name}:  dt = {dt:.4e} s   T_cold = {s["T_cold"].min():.3g}-{s["T_cold"].max():.3g} eV   '
          f'n_re max = {s["n_re"].max():.3e} m^-3')
    if P_RE / NP_HOT > p_th / 5:
        print(f'    WARNING: p_th = {p_th:.2e} at the coldest radius is not resolved by dp = {P_RE/NP_HOT:.2e}; '
              f'raise LEVEL (needs ~{int(np.ceil(5 * P_RE / p_th))} points in p)')
    print(f'    Wrote {settings_file}')

with open(os.path.join(OUT_DIR, 'petsc_num_timesteps.txt'), 'w') as fh:
    fh.write('1\n')

print()
print(f'Run:  cd {OUT_DIR} && ./dreami dream_settings_kinetic_<snapshot>_L{LEVEL}.h5')
