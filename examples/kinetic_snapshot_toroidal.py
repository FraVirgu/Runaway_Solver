#!/usr/bin/env python3
#
# Kinetic-only DREAM runs started from the snapshots in DATASET/snapshots/,
# as examples/kinetic_from_snapshots.py, but in TOROIDAL geometry: DREAM's
# analytic toroidal field (RadialGrid.TYPE_ANALYTIC_TOROIDAL), with the
# shape fitted to the ITER equilibrium of the source simulation, and a
# radial resolution that can be raised above the snapshot's.
#
# Geometry
# --------
# The source runs use the numerical ITER equilibrium ITER_H26_LUKE.h5, which
# is not available; its flux surfaces are, in every state.h5
# (grid/eq/RMinusR0, ZMinusZ0 on 80 poloidal angles, on the cell centres and
# faces), together with grid/R0 (magnetic axis), grid/geometry/GR0 (G/R0 =
# R B_phi / R0), Bmin and Bmax. DREAM's analytic model is
#
#   R = R0 + Delta(r) + r cos(theta + delta(r) sin(theta)),   Z = kappa(r) r sin(theta)
#   B_tor = (G/R0) / (R/R0),   B_pol = |grad r| psi'(r) / (2 pi R/R0)
#
# and each flux surface of the snapshot gives
#
#   r      half width (R_out - R_in)/2: the radial coordinate of the model
#          (the source code uses the outboard distance R_out - R0 instead:
#          the profiles are mapped surface by surface)
#   Delta  Shafranov shift (R_out + R_in)/2 - R0
#   kappa  elongation (Z_top - Z_bottom) / (2 r)
#   delta  triangularity, least-squares fit of the whole surface
#   G/R0   from grid/geometry/GR0
#   psi'   from B_min (outboard midplane, theta = 0, where |grad r| =
#          1/(1 + Delta')): B_pol = sqrt(B_min^2 - B_tor^2)
#
# The fit is checked against the snapshot's B_max (inboard midplane, not
# used in the fit) and printed. The model is up-down symmetric: the X-point
# asymmetry of the outermost surfaces is averaged out.
#
# In toroidal geometry DREAM adds the trapped-region boundary condition
# (FVM::BC::PXiInternalTrapping) to f_hot and f_re.
#
# Radial resolution
# -----------------
# NR (default: the snapshot's, 11) radial cells, uniform in r between the
# axis and the last flux surface (with NR = the snapshot's, the snapshot's
# flux surfaces themselves). The profiles (E, T_cold, n_cold, n_i, n_re) are
# interpolated linearly in r (constant beyond the outermost centres). A
# larger NR gives more independent blocks to the radial block Jacobi
# (-dream rbj: up to NR ranks without splitting a radius).
#
# Everything else (grids, transport, initial condition, solver) is as in
# kinetic_from_snapshots.py, with the same environment variables:
#   LEVEL (2), ADV (quick), DBB (1e-4), AR_FACTOR (10), RE_PT (10),
#   RE_XIW (0.05), NT_CQ (1), NT_INJ (1), SNAPSHOT (all)
# and, for the geometry,
#   NR      radial cells (default: snapshot's)
#   NTHETA  poloidal points of the bounce averages (default 20, as the source)
#
# Files: $DREAM_ROOT/build/iface/dream_settings_toroidal_<snapshot>_nr<NR>_L<LEVEL>.h5
# (output_toroidal_..._L<LEVEL>.h5). run_memory_snapshots.sh runs them with
# PREFIX=toroidal.
#
# Run as
#
#   $ ./kinetic_snapshot_toroidal.py                        # all snapshots, snapshot Nr
#   $ NR=44 ./kinetic_snapshot_toroidal.py                  # 44 radial cells
#   $ SNAPSHOT=CQ_tf NR=22 LEVEL=1 ./kinetic_snapshot_toroidal.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_toroidal_CQ_tf_nr11_L2.h5
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
import DREAM.Settings.RadialGrid as RadialGrid
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport

from extract_snapshots import SNAPSHOTS

mc2 = 510998.95             # electron rest energy [eV]

# =====================================================================
# Numerical parameters (as in kinetic_from_snapshots.py)
# =====================================================================
P_RE  = 1.0                 # hot/runaway grid interface
P_MAX = 60.0

LEVEL = int(os.environ.get('LEVEL', 2))
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
}[os.environ.get('ADV', 'quick').lower()]

DBB       = float(os.environ.get('DBB', 1e-4))       # Rechester-Rosenbluth dB/B (0: off)
AR_FACTOR = float(os.environ.get('AR_FACTOR', 10.0)) # A_r(r) = AR_FACTOR * E(r) [m/s] (0: off)
RE_PT     = float(os.environ.get('RE_PT', 10.0))     # initial f_re: e-folding momentum
RE_XIW    = float(os.environ.get('RE_XIW', 0.05))    # initial f_re: pitch width

NT_CQ  = int(os.environ.get('NT_CQ', 1))
NT_INJ = int(os.environ.get('NT_INJ', 1))

# Geometry
NR     = int(os.environ['NR']) if os.environ.get('NR') else None   # None: snapshot's
NTHETA = int(os.environ.get('NTHETA', 20))

def num_timesteps(name):
    return NT_CQ if name.startswith('CQ') else NT_INJ


p_re_init  = np.linspace(P_RE, P_MAX, 500)
xi_re_init = np.linspace(-1.0, 1.0,    401)


def addRadialAdvection(tr, ar, r):
    """
    Prescribe the radial advection ar(r) [m/s] (constant in t, p, xi) on the
    kinetic transport settings tr, keeping a Rechester-Rosenbluth dB/B set
    before (see kinetic_from_snapshots.py).
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


# =====================================================================
# Geometry: fit of DREAM's analytic toroidal model to the snapshot
# =====================================================================
def surface_shape(dR, dZ):
    """
    Shape of the flux surfaces given as R - R0, Z - Z0 [ntheta, nsurf]:
    half width r, Shafranov shift Delta, elongation kappa, vertical centre
    Zc and the least-squares triangularity delta of each surface (kappa,
    delta = nan on a degenerate surface, r = 0).
    """
    Rout, Rin = dR.max(axis=0), dR.min(axis=0)
    Ztop, Zbot = dZ.max(axis=0), dZ.min(axis=0)
    r = (Rout - Rin) / 2
    Delta = (Rout + Rin) / 2
    Zc = (Ztop + Zbot) / 2

    kappa = np.full(r.shape, np.nan)
    delta = np.full(r.shape, np.nan)
    th = np.linspace(0, 2*np.pi, 721)[:-1]
    deltas = np.linspace(-0.4, 0.9, 1301)
    for j in np.nonzero(r > 0)[0]:
        kappa[j] = (Ztop[j] - Zbot[j]) / (2 * r[j])
        x = (dR[:, j] - Delta[j]) / r[j]                  # data, normalised
        z = (dZ[:, j] - Zc[j]) / (r[j] * kappa[j])
        best = (np.inf, 0.0)
        # coarse-to-fine search of the delta that brings the model curve
        # closest to the data points (mean squared distance)
        for grid in (deltas, None):
            if grid is None:
                d0 = best[1]
                grid = np.linspace(d0 - 2e-3, d0 + 2e-3, 81)
            for d in grid:
                xm = np.cos(th + d * np.sin(th))
                zm = np.sin(th)
                dist = np.min((x[:, None] - xm[None, :])**2 + (z[:, None] - zm[None, :])**2, axis=1)
                err = dist.mean()
                if err < best[0]:
                    best = (err, d)
        delta[j] = best[1]
    return r, Delta, kappa, delta, Zc


def fit_geometry(state):
    """
    Analytic toroidal geometry fitted to the equilibrium stored in the
    snapshot (state.h5). Returns a dict with R0, the half-width coordinate
    of the snapshot faces (r_f) and centres (r_c), the shape profiles on
    r_f, G/R0 and psi_p0 on their own radial grids, and the B_max check.
    """
    with h5py.File(state, 'r') as f:
        R0  = f['grid/R0'][0]
        eq  = f['grid/eq']
        dRf, dZf = eq['RMinusR0_f'][:], eq['ZMinusZ0_f'][:]
        dRc, dZc = eq['RMinusR0'][:],   eq['ZMinusZ0'][:]
        g = f['grid/geometry']
        GR0  = np.ravel(g['GR0'][:])
        Bmin = np.ravel(g['Bmin'][:])
        Bmax = np.ravel(g['Bmax'][:])

    r_f, Delta_f, kappa_f, delta_f, _ = surface_shape(dRf, dZf)
    r_c, Delta_c, _, _, _             = surface_shape(dRc, dZc)

    # On axis: no shift, no triangularity, the elongation of the first surface
    Delta_f[0] = 0.0
    delta_f[0] = 0.0
    kappa_f[0] = kappa_f[1]

    # psi'(r) at the centres from B_min = B(theta = 0):
    #   R/R0 = 1 + (Delta + r)/R0,  |grad r| = 1/(1 + Delta'),
    #   B_pol^2 = B_min^2 - (G/R0 / (R/R0))^2,  psi' = 2 pi (R/R0) B_pol (1 + Delta')
    DeltaP_c = np.diff(Delta_f) / np.diff(r_f)
    ROut_c = 1 + (Delta_c + r_c) / R0
    Bpol_c = np.sqrt(np.maximum(Bmin**2 - (GR0 / ROut_c)**2, 0.0))
    psiP_c = 2 * np.pi * ROut_c * Bpol_c * (1 + DeltaP_c)

    # psi = int psi' dr from the axis (psi'(0) = 0), extended linearly to the edge
    r_psi = np.concatenate(([0.0], r_c, [r_f[-1]]))
    psiP  = np.concatenate(([0.0], psiP_c))
    psi   = np.concatenate(([0.0], np.cumsum(0.5 * (psiP[1:] + psiP[:-1]) * np.diff(r_psi[:-1]))))
    psi   = np.append(psi, psi[-1] + psiP_c[-1] * (r_f[-1] - r_c[-1]))

    # G/R0 extended linearly to the axis and the edge
    def ext(x, y, x0):
        return y[0] + (y[1] - y[0]) * (x0 - x[0]) / (x[1] - x[0])
    r_G = np.concatenate(([0.0], r_c, [r_f[-1]]))
    G   = np.concatenate(([ext(r_c, GR0, 0.0)], GR0, [ext(r_c[::-1], GR0[::-1], r_f[-1])]))

    # Check: B_max = B(theta = pi), where |grad r| = 1/(1 - Delta')
    RIn_c = 1 + (Delta_c - r_c) / R0
    Bmax_model = np.hypot(GR0 / RIn_c, psiP_c * (1 - DeltaP_c)**-1 / (2 * np.pi * RIn_c))

    return dict(R0=R0, r_f=r_f, r_c=r_c, Delta=Delta_f, kappa=kappa_f, delta=delta_f,
                r_G=r_G, G=G, r_psi=r_psi, psi=psi,
                Bmin=Bmin, Bmax=Bmax, Bmax_model=Bmax_model)


def load_snapshot(folder):
    state = os.path.join(folder, 'state.h5')
    with h5py.File(state, 'r') as f:
        s = {q: f['eqsys'][q][0] for q in ('E_field', 'T_cold', 'n_cold', 'n_re', 'n_i')}
        s['wall'] = f['settings/radialgrid/wall_radius'][0]

        ni = f['settings/eqsys/n_i']
        s['ion_names']    = [n for n in ni['names'][()].tobytes().decode().split(';') if n]
        s['ion_Z']        = ni['Z'][:]
        s['ion_isotopes'] = ni['isotopes'][:]
    s['geo'] = fit_geometry(state)
    return s


def radial_grid(geo):
    """Faces and centres of the simulation grid in the model coordinate r."""
    nr0 = geo['r_c'].size
    nr = NR or nr0
    if nr == nr0:
        r_f = geo['r_f'].copy()          # the snapshot's own flux surfaces
    else:
        r_f = np.linspace(0.0, geo['r_f'][-1], nr + 1)
    return r_f, 0.5 * (r_f[1:] + r_f[:-1])


def on_grid(r_new, r_old, y):
    """Linear interpolation in r (constant beyond the end points), along the last axis."""
    y = np.asarray(y)
    if y.ndim == 1:
        return np.interp(r_new, r_old, y)
    return np.stack([np.interp(r_new, r_old, row) for row in y.reshape(-1, y.shape[-1])]).reshape(
        y.shape[:-1] + (r_new.size,))


def f_re_init(n_re, xi0):
    """Runaway beam along xi0 = +-1 with density n_re(r), shape (nr, nxi, np)."""
    P, XI = np.meshgrid(p_re_init, xi_re_init)
    shape = np.exp(-(P - P_RE) / RE_PT) * np.exp(-(1 - xi0 * XI) / RE_XIW)
    # n = 2 pi int int p^2 f dp dxi
    norm = 2 * np.pi * np.trapezoid(np.trapezoid(P**2 * shape, p_re_init, axis=1), xi_re_init)
    return n_re[:, np.newaxis, np.newaxis] * (shape / norm)[np.newaxis, :, :]


def make_settings(name, s, dt):
    geo = s['geo']
    r_f, r = radial_grid(geo)

    # Snapshot profiles (on the snapshot's surfaces, coordinate r_c) on the new grid
    E      = on_grid(r, geo['r_c'], s['E_field'])
    T_cold = on_grid(r, geo['r_c'], s['T_cold'])
    n_cold = on_grid(r, geo['r_c'], s['n_cold'])
    n_re   = on_grid(r, geo['r_c'], s['n_re'])
    n_i    = on_grid(r, geo['r_c'], s['n_i'])

    ds = DREAMSettings()

    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    # -----------------------------------------------------------------
    # Fixed background from the snapshot
    # -----------------------------------------------------------------
    ds.eqsys.E_field.setPrescribedData(E, radius=r)
    ds.eqsys.T_cold.setPrescribedData(T_cold, radius=r)
    ds.eqsys.n_cold.setPrescribedData(n_cold, radius=r)

    # n_i: consecutive blocks of Z+1 charge states per species, one time point
    i0 = 0
    for ion, Z, iso in zip(s['ion_names'], s['ion_Z'], s['ion_isotopes']):
        ds.eqsys.n_i.addIon(name=ion, Z=int(Z), isotope=int(iso), iontype=Ions.IONS_PRESCRIBED,
                            n=n_i[i0:i0+Z+1, np.newaxis, :], r=r, t=np.array([0.0]))
        i0 += Z + 1

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

    # -----------------------------------------------------------------
    # Hot-tail grid: [0, P_RE], Maxwellian initial condition
    # -----------------------------------------------------------------
    ds.hottailgrid.setNp(NP_HOT)
    ds.hottailgrid.setNxi(NXI_HOT)
    ds.hottailgrid.setPmax(P_RE)

    ds.eqsys.f_hot.setInitialProfiles(n0=n_cold, T0=T_cold, rn0=r, rT0=r)
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

    if np.any(n_re > 0):
        # E > 0 accelerates toward xi = +1
        xi0 = 1.0 if np.mean(E) >= 0 else -1.0
        ds.eqsys.f_re.setInitialValue(f_re_init(n_re, xi0), r=r, p=p_re_init, xi=xi_re_init)
    ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
    ds.eqsys.f_re.setAdvectionInterpolationMethod(ADV_INTERP)

    # -----------------------------------------------------------------
    # Geometry: analytic toroidal, fitted to the snapshot equilibrium
    # -----------------------------------------------------------------
    ds.radialgrid.setType(RadialGrid.TYPE_ANALYTIC_TOROIDAL)
    ds.radialgrid.setMajorRadius(geo['R0'])
    ds.radialgrid.setCustomGridPoints(r_f)
    ds.radialgrid.setWallRadius(max(s['wall'], r_f[-1]))
    ds.radialgrid.setNtheta(NTHETA)
    ds.radialgrid.setShaping(
        psi=geo['psi'], rpsi=geo['r_psi'],
        GOverR0=geo['G'], rG=geo['r_G'],
        Delta=geo['Delta'], rDelta=geo['r_f'],
        delta=geo['delta'], rdelta=geo['r_f'],
        kappa=geo['kappa'], rkappa=geo['r_f'])

    # -----------------------------------------------------------------
    # Radial transport
    # -----------------------------------------------------------------
    if DBB > 0:
        ds.eqsys.f_hot.transport.setMagneticPerturbation(DBB)
        ds.eqsys.f_hot.transport.setBoundaryCondition(Transport.BC_F_0)
        ds.eqsys.f_re.transport.setMagneticPerturbation(DBB)
        ds.eqsys.f_re.transport.setBoundaryCondition(Transport.BC_F_0)

    if AR_FACTOR != 0:
        ar = AR_FACTOR * E
        addRadialAdvection(ds.eqsys.f_hot.transport, ar, r)
        addRadialAdvection(ds.eqsys.f_re.transport, ar, r)

    # -----------------------------------------------------------------
    # Solver: the snapshot dt, in num_timesteps(name) steps
    # -----------------------------------------------------------------
    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.solver.preconditioner.setEnabled(False)
    ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_RADIAL_BJ)

    ds.other.include('fluid', 'nu_s', 'nu_D')

    ds.timestep.setTmax(dt)
    ds.timestep.setNt(num_timesteps(name))

    ds.output.setTiming(stdout=True, file=True)
    return ds, r, n_re, T_cold


only = os.environ.get('SNAPSHOT')
folders = sorted(f for f in glob.glob(os.path.join(SNAP_DIR, '*')) if os.path.isdir(f))
if only:
    folders = [f for f in folders if os.path.basename(f) == only]
if not folders:
    sys.exit(f'No snapshots found in {SNAP_DIR}; run DATASET/extract_snapshots.py first.')

os.makedirs(OUT_DIR, exist_ok=True)
print('=' * 70)
print(f'Kinetic runs from DATASET snapshots, analytic toroidal geometry   [LEVEL = {LEVEL}]')
print(f'  hot grid  Np = {NP_HOT}  Nxi = {NXI_HOT}   dp = {P_RE/NP_HOT:.2e}')
print(f'  RE grid   Np = {NP_RE}  Nxi = {NXI_RE}')
print(f'  Nr = {NR or "snapshot"}   Ntheta = {NTHETA}')
print(f'  dB/B = {DBB:.1e}   A_r = {AR_FACTOR:g} E(r)   advection {os.environ.get("ADV", "quick")}')
print('=' * 70)

geometry_printed = False
for folder in folders:
    name = os.path.basename(folder)
    s = load_snapshot(folder)
    dt = snapshot_dt(name)
    geo = s['geo']

    # The equilibrium is the same in every snapshot: print the fit once
    if not geometry_printed:
        print(f'Geometry fitted to the equilibrium of {name}: R0 = {geo["R0"]:.4f} m, '
              f'a = {geo["r_f"][-1]:.4f} m (half width of the last surface)')
        print(f'  {"r_f [m]":>8}  {"Delta [m]":>9}  {"kappa":>6}  {"delta":>6}')
        for rf, D, k, d in zip(geo['r_f'], geo['Delta'], geo['kappa'], geo['delta']):
            print(f'  {rf:8.4f}  {D:9.4f}  {k:6.3f}  {d:6.3f}')
        err = np.abs(geo['Bmax_model'] / geo['Bmax'] - 1)
        print(f'  check B_max (inboard midplane, not fitted): model vs snapshot, max relative '
              f'difference {err.max():.1e} (at r = {geo["r_c"][err.argmax()]:.3f} m)')
        print(f'  trapped fraction at the edge ~ sqrt(1 - B_min/B_max) = '
              f'{np.sqrt(1 - geo["Bmin"][-1] / geo["Bmax"][-1]):.2f}')
        geometry_printed = True

    ds, r, n_re, T_cold = make_settings(name, s, dt)
    TAG = f'toroidal_{name}_nr{r.size}_L{LEVEL}'
    ds.output.setFilename(f'output_{TAG}.h5')
    settings_file = os.path.join(OUT_DIR, f'dream_settings_{TAG}.h5')
    ds.save(settings_file)

    p_th = np.sqrt(2 * T_cold.min() / mc2)
    nt = num_timesteps(name)
    print(f'{name}:  Nr = {r.size}   dt = {dt:.4e} s in {nt} step(s)   '
          f'T_cold = {T_cold.min():.3g}-{T_cold.max():.3g} eV   n_re max = {n_re.max():.3e} m^-3')
    if P_RE / NP_HOT > p_th / 5:
        print(f'    WARNING: p_th = {p_th:.2e} at the coldest radius is not resolved by dp = {P_RE/NP_HOT:.2e}; '
              f'raise LEVEL (needs ~{int(np.ceil(5 * P_RE / p_th))} points in p)')
    print(f'    Wrote {settings_file}')

print()
print(f'Run:  cd {OUT_DIR} && ./dreami dream_settings_toroidal_<snapshot>_nr<NR>_L{LEVEL}.h5')
print(f'      PREFIX=toroidal sbatch run_memory_snapshots.sh   (memory / speed-up)')
