#!/usr/bin/env python3
#
# MPI check of the parallel dreami at POST-TQ / EARLY-CQ conditions, i.e.
# the regime of the JOREK-DREAM coupling (Svensson thesis, Fig. 4.1):
#
#   T_e ~ 1-10 eV,  n_e ~ 0.4-1.6e20 m^-3,  Z_eff ~ 1-2,  E ~ 30 V/m
#
# Companion of mpi_check_code_runaway.py. DREAM's physics tests have CODE
# reference data only at T >= 100 eV, so their references are TRANSFERRED to
# post-TQ temperature and density with exact scaling laws. Three families:
#
# 1. conductivity  (from tests/physics/code_conductivity, f_hot only)
#      Non-relativistically sigma = sigma_hat(Z) T^{3/2} / (Z lnL), independent
#      of n. The reference at (T', n') is the CODE value at 100 eV carried
#      over with DREAM's Braams-Karney formula, which contains exactly that
#      scaling plus the relativistic correction (0.07 % at 100 eV):
#
#        sigma_ref(T', n', Z) = sigma_CODE(100 eV, Z) * BK(n', T', Z) / BK(5e19, 100 eV, Z)
#
#      CODE (test-particle operator, like DREAM) gives the REDUCED
#      conductivity sigma_CODE / sigma_BK = 0.508, 0.645, 0.763 for Z = 1, 2, 4,
#      matching thesis Eq. 2.3.6, (1 - 1.406/(1.888+Z)) = 0.513, 0.639, 0.761,
#      which is printed alongside as a second reference.
#      Residual non-similarity: DREAM's thermal lnL_ee and lnL_ei differ
#      from the Wesson lnL by fixed constants, so their ratio drifts by
#      ~1 % between 100 eV and 1 eV. Tolerance COND_TOL = 2 % (test: 1 %).
#
# 2. dreicer       (from tests/physics/code_runaway, f_hot only)
#      The Dreicer rate is n nu_ee F(E/E_D, Z) up to relativistic corrections.
#      Each CODE point at 100 eV is moved to (T', n') at the SAME E/E_D:
#
#        E'      = E * E_D(T', n') / E_D(100 eV, 5e19)
#        rr_ref' = rr_CODE * CH(T', n', E') / CH(100 eV, 5e19, E)
#
#      with CH the Connor-Hastie rate (DREAM.Formulas), which supplies the
#      n nu_ee scaling and the relativistic correction (the latter is small,
#      so errors in CH enter only at second order). Tolerance DREI_TOL = 8 %
#      (test: 5 %; +~3 % from the lnL offsets in the exponent E_D/4E).
#      NOTE: at the thesis field (35 V/m at 8 eV, E/E_D ~ 5e-3) the Dreicer
#      rate is ~e^-50, i.e. zero. This family therefore uses E' of a few
#      hundred V/m: it tests the post-TQ COLLISION OPERATOR (cold, dense,
#      stiff), not the thesis field.
#
# 3. avalanche     (from tests/physics/DREAM_avalanche, needs n_re)
#      DREAM_avalanche is already a post-TQ test: T = 10 eV, D, neutral Ne/Ar,
#      E/E_c,tot = 5-100. Its points are checked against the stored reference
#      data (test tolerance 5 %). An extra THESIS point (T = 8.35 eV,
#      n_e = 1.58e20, ionised Ne for Z_eff = 1.5, E = 35.5 V/m) has no stored
#      data and is compared with the analytic Hesslow growth rate, report-only
#      (formula accuracy ~10-20 %; WARN above AVA_ANALYTIC_TOL).
#      The avalanche source is proportional to n_re, so this family is run
#      WITHOUT -dream_kinetic_only (n_re must be in the system).
#
# For every point and every number of ranks in NPROCS, as in the companion:
#   reference   rate / conductivity on 1 rank vs the (transferred) reference
#   parallel    run on P ranks vs 1 rank: f_hot and the checked quantity
#
# Run as
#
#   $ ./mpi_check_postTQ.py                                   # conductivity + dreicer
#   $ FAMILIES="conductivity dreicer avalanche" ./mpi_check_postTQ.py
#   $ DRY_RUN=1 ./mpi_check_postTQ.py                         # settings + references only
#   $ FAMILIES="conductivity dreicer" NPROCS="1 4 8" ./mpi_check_postTQ.py
#   $ T_POST="1.3 3 8.35" N_POST=1.58e20 ./mpi_check_postTQ.py
#
# Environment (beyond those of mpi_check_code_runaway.py: DREAM_ROOT, NPROCS,
# MPIRUN, FTOL, RTOL, MINREL, OUTDIR):
#   FAMILIES     subset of "conductivity dreicer avalanche" (default: first two)
#   COND_ADV     centred (test default) or tcdf, for the conductivity family
#   T_POST       post-TQ temperatures [eV]       (default "1.3 3 8.35": Fig. 4.1 edge..core)
#   N_POST       post-TQ density [m^-3]          (default 1.58e20: Fig. 4.1 core)
#   Z_COND       charges for conductivity        (default "1 2 4")
#   T_DREICER    temperature for Dreicer [eV]    (default 8.35)
#   AVA_POINTS   "all", "none" or index tuples "iE,iD0,iD1,iAr,iNe ..."
#                (default: the three E with n_D+ = 1e21, n_Ne = 1e20, rest low)
#   AVA_THESIS   1/0, include the thesis avalanche point (default 1)
#   PETSC_KIN    PETSc options for f_hot-only runs (default "-dream_kinetic_only -dream_scale diag")
#   PETSC_FULL   PETSc options for avalanche runs   (default "-dream_scale diag")
#
# ###################################################################

import copy
import os
import subprocess
import sys
import numpy as np

DREAM_ROOT = os.path.expanduser(os.environ.get(
    'DREAM_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'DREAM')))
TESTDIR = os.path.join(DREAM_ROOT, 'tests', 'physics')

sys.path.append(os.path.join(DREAM_ROOT, 'py'))
sys.path.append(TESTDIR)                                    # 'dreamtests'
for t in ('code_runaway', 'code_conductivity', 'DREAM_avalanche'):
    sys.path.append(os.path.join(TESTDIR, t))

import DREAM.Formulas as F
from DREAM.DREAMOutput import DREAMOutput
from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.IonSpecies as IonSpecies
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import code_runaway
import code_conductivity
import DREAM_avalanche

DREAMI   = os.path.join(DREAM_ROOT, 'build', 'iface', 'dreami')
NPROCS   = [int(p) for p in os.environ.get('NPROCS', '1 2 4').split()]
MPIRUN   = os.environ.get('MPIRUN', 'mpirun -n').split()
FTOL     = float(os.environ.get('FTOL', 1e-6))
RTOL     = float(os.environ.get('RTOL', 1e-3))
MINREL   = float(os.environ.get('MINREL', 1e-6))
OUTDIR   = os.path.abspath(os.environ.get('OUTDIR', 'mpi_check_postTQ'))
DRY_RUN  = bool(int(os.environ.get('DRY_RUN', 0)))

# avalanche is opt-in: it needs n_re in the system (full mode), which does
# not yet run in the parallel build (initializer skipped -> NaN/inf; LU is
# serial-only; AMG on the fluid rows breaks down).
FAMILIES  = os.environ.get('FAMILIES', 'conductivity dreicer').split()
T_POST    = [float(x) for x in os.environ.get('T_POST', '1.3 3 8.35').split()]
N_POST    = float(os.environ.get('N_POST', 1.58e20))
Z_COND    = [float(x) for x in os.environ.get('Z_COND', '1 2 4').split()]
T_DREICER = float(os.environ.get('T_DREICER', 8.35))
PETSC_KIN  = os.environ.get('PETSC_KIN',  '-dream_kinetic_only -dream_scale diag')
PETSC_FULL = os.environ.get('PETSC_FULL', '-dream_scale diag')

COND_TOL = float(os.environ.get('COND_TOL', 2e-2))
DREI_TOL = float(os.environ.get('DREI_TOL', 8e-2))
AVA_TOL  = 5e-2                                  # DREAM_avalanche's own tolerance
AVA_ANALYTIC_TOL = float(os.environ.get('AVA_ANALYTIC_TOL', 0.2))

T_REF, N_REF = 100.0, 5e19                       # CODE point used for the transfer

if NPROCS[0] != 1:
    sys.exit('NPROCS must start with 1 (the serial reference).')

# LINEAR_SOLVER_AMG exists only in the parallel DREAM branch; upstream DREAM
# keeps its default solver (used e.g. for DRY_RUN against an upstream tree).
LINSOLVER = getattr(Solver, 'LINEAR_SOLVER_AMG', None)


def set_amg(ds):
    if LINSOLVER is not None:
        ds.solver.setLinearSolver(LINSOLVER)


# =====================================================================
# Running dreami
# =====================================================================
def run_dreami(ds, tag, nprocs, petsc):
    """Save settings, run dreami on 'nprocs' ranks, return the output path or None."""
    ds.output.setFilename(f'output_{tag}.h5')
    ds.save(os.path.join(OUTDIR, f'settings_{tag}.h5'))
    if DRY_RUN:
        return None
    env = dict(os.environ)
    env['PETSC_OPTIONS'] = env.get('PETSC_OPTIONS', '') + ' ' + petsc
    log = os.path.join(OUTDIR, f'log_{tag}.txt')
    with open(log, 'w') as fh:
        cmd = ([DREAMI] if nprocs == 1 and not petsc else MPIRUN + [str(nprocs), DREAMI])
        subprocess.run(cmd + [f'settings_{tag}.h5'], cwd=OUTDIR, env=env,
                       stdout=fh, stderr=subprocess.STDOUT)
    out = os.path.join(OUTDIR, f'output_{tag}.h5')
    if not os.path.isfile(out):          # dreami can exit 0 after a failed step
        print(f'    np {nprocs:2d}   dreami FAILED, see {log}')
        return None
    return out


# =====================================================================
# Family 1: conductivity
# =====================================================================
def conductivity_points():
    Tc, Zc, sc = code_conductivity.loadCODE(os.path.join(TESTDIR, 'code_conductivity',
                                                         'CODE-conductivities.mat'))
    pts = []
    for Z in Z_COND:
        hit = np.argwhere((np.isclose(Tc, T_REF)) & (np.isclose(Zc, Z)))
        if hit.size == 0:
            print(f'  (no CODE conductivity at T = {T_REF} eV, Z = {Z}: skipped)')
            continue
        s_code = sc[tuple(hit[0])]
        bk_ref = np.ravel(F.evaluateBraamsConductivity(N_REF, T_REF, Z))[0]
        for T in T_POST:
            bk = np.ravel(F.evaluateBraamsConductivity(N_POST, T, Z))[0]
            # The test-particle operator (as in DREAM and in this CODE run)
            # gives a reduced conductivity; thesis Eq. 2.3.6 fits it as
            # sigma_kineq = (1 - 1.406/(1.888 + Z)) sigma_BK. Printed as a
            # second, independent reference.
            kin = (1 - 1.406 / (1.888 + Z)) * bk
            pts.append(dict(T=T, Z=Z, ref=s_code * bk / bk_ref,
                            label=f'T = {T:g} eV, n = {N_POST:.2e}, Z = {Z:g}',
                            note=f'CODE(100 eV) {s_code:.4e} x BK ratio {bk/bk_ref:.4e}; '
                                 f'Eq. 2.3.6 gives {kin:.4e}'))
    return pts


def conductivity_case(p, nprocs):
    ds = code_conductivity.gensettings(T=p['T'], Z=p['Z'], n=N_POST)
    # -dream_kinetic_only needs n_cold prescribed (EquationSystem.cpp). This
    # matches the test: the bulk is counted as cold (n_hot is p > 7 p_th).
    ds.eqsys.n_cold.setPrescribedData(N_POST)
    if os.environ.get('COND_ADV', 'centred').lower() == 'tcdf':
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_TCDF)
    # same normalised simulation time as the test at n = 5e19
    ds.timestep.setTmax(3.5e-3 * (p['T'] / 1e3)**1.5 * N_REF / N_POST)
    set_amg(ds)
    tag = f'cond_T{p["T"]:g}_Z{p["Z"]:g}_np{nprocs}'
    out = run_dreami(ds, tag, nprocs, PETSC_KIN)
    if out is None:
        return None
    do = DREAMOutput(out)
    f = np.array(do.eqsys.f_hot.data)
    j = do.eqsys.f_hot.currentDensity(t=-1)[0, 0]
    E = do.eqsys.E_field[-1, 0]
    do.close()
    return f, j / E, True


# =====================================================================
# Family 2: Dreicer
# =====================================================================
def dreicer_points():
    Tr, Er, rr = code_runaway.loadCODE(os.path.join(TESTDIR, 'code_runaway', 'CODE-rates.mat'))
    jT = int(np.argmin(np.abs(Tr[0] - T_REF)))
    EDr, EDp = F.getED(T_REF, N_REF), F.getED(T_DREICER, N_POST)
    pts = []
    for i in range(Er.shape[0]):
        E, r = Er[i, jT], rr[i, jT]
        Ep = E * EDp / EDr
        chr_ = F.getConnorHastieRunawayRate(T_REF, N_REF, E, 1)
        chp  = F.getConnorHastieRunawayRate(T_DREICER, N_POST, Ep, 1)
        ref  = r * chp / chr_ if chr_ > 0 else 0.0
        pts.append(dict(T=T_DREICER, E=Ep, ref=ref,
                        label=f'T = {T_DREICER:g} eV, n = {N_POST:.2e}, E = {Ep:.4g} V/m '
                              f'(E/E_D = {E/EDr:.4f})',
                        note=f'CODE(100 eV, {E:.4g} V/m) {r:.3e} x CH ratio '
                             f'{(chp/chr_ if chr_ > 0 else 0):.4e}'))
    return pts


def dreicer_case(p, nprocs):
    ds = code_runaway.gensettings(T=p['T'], E=p['E'], n=N_POST)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_TCDF)
    set_amg(ds)
    tag = f'drei_T{p["T"]:g}_E{p["E"]:.4g}_np{nprocs}'
    out = run_dreami(ds, tag, nprocs, PETSC_KIN)
    if out is None:
        return None
    do = DREAMOutput(out)
    f = np.array(do.eqsys.f_hot.data)
    n = do.eqsys.f_hot.density()[:, 0]
    t = np.array(do.grid.t)
    do.close()
    # as in mpi_check_code_runaway.py: loss through p_max over the last step
    rr = -(n[-1] - n[-2]) / (t[-1] - t[-2])
    return f, rr, abs(n[-1] - n[-2]) / n[-2] >= MINREL


# =====================================================================
# Family 3: avalanche
# =====================================================================
def thesis_avalanche_settings(T=8.35, ne=1.58e20, Zeff=1.5, E=35.5):
    """DREAM_avalanche.gensettings with the thesis plasma: D + IONISED Ne."""
    Z = 10
    nNe = ne * (Zeff - 1) / (Z * (Z - 1))
    nD  = ne * (Z - Zeff) / (Z - 1)
    EcTot = F.getEc(T, ne) * 1.0           # n_tot = n_e (fully ionised)
    pcTot = 1 / np.sqrt(E / EcTot - 1)
    pMax = min(20 * pcTot, 10)

    ds = DREAMSettings()
    ds.collisions.lnlambda = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    ds.collisions.collfreq_mode = Collisions.COLLFREQ_MODE_SUPERTHERMAL
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.eqsys.E_field.setPrescribedData(E)
    ds.eqsys.n_i.addIon(name='D',  Z=1,  n=nD,  iontype=IonSpecies.IONS_PRESCRIBED_FULLY_IONIZED)
    ds.eqsys.n_i.addIon(name='Ne', Z=Z,  n=nNe, iontype=IonSpecies.IONS_PRESCRIBED_FULLY_IONIZED)
    ds.eqsys.T_cold.setPrescribedData(T)
    ds.eqsys.f_hot.setInitialProfiles(n0=0.01, T0=1e5)
    ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=FHot.AD_INTERP_TCDF)
    ds.eqsys.f_hot.setBoundaryCondition(FHot.BC_F_0)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_KINETIC, pCutAvalanche=0.01)
    ds.eqsys.n_re.setEceff(Eceff=Runaways.COLLQTY_ECEFF_MODE_SIMPLE)
    ds.eqsys.n_re.setInitialProfile(density=1)
    ds.eqsys.f_hot.enableIonJacobian(False)
    ds.hottailgrid.setNxi(15)
    ds.hottailgrid.setNp(60)
    ds.hottailgrid.setPmax(pMax)
    ds.runawaygrid.setEnabled(False)
    ds.radialgrid.setB0(1e-6)
    ds.radialgrid.setMinorRadius(0.1)
    ds.radialgrid.setWallRadius(0.1)
    ds.radialgrid.setNr(1)
    me, c, e = 9.1093837015e-31, 2.99792458e8, 1.602176634e-19
    ds.timestep.setTmax(30 * me * c / ((E - EcTot) * e))
    ds.timestep.setNt(DREAM_avalanche.nTimeSteps)
    ds.solver.setType(Solver.NONLINEAR)
    ds.solver.tolerance.set(reltol=1e-4)
    ds.other.include('fluid')
    return ds, E / EcTot


def avalanche_points():
    import h5py
    with h5py.File(os.path.join(TESTDIR, 'DREAM_avalanche', 'DREAM-avalanche-data.h5'), 'r') as f:
        G = f['GammaNum'][:]
        EE, nDs, nZs = f['EOverEcTot'][:], f['nD'][:], f['nZ'][:]
    sel = os.environ.get('AVA_POINTS', '')
    if sel == 'none':
        idx = []
    elif sel == 'all':
        idx = [tuple(i) for i in np.ndindex(G.shape)]
    elif sel:
        idx = [tuple(int(k) for k in s.split(',')) for s in sel.split()]
    else:
        idx = [(i, 1, 0, 0, 1) for i in range(len(EE))]       # D+ 1e21, Ne 1e20
    pts = []
    for (i, j, k, m, n) in idx:
        pts.append(dict(kind='ref', id=f'E{i}D{j}{k}Ar{m}Ne{n}', args=dict(EOverEcTot=EE[i], nD0=nDs[j], nD1=nDs[k],
                                              nAr=nZs[m], nNe=nZs[n]),
                        ref=G[i, j, k, m, n],
                        label=f'T = 10 eV, E/Ec,tot = {EE[i]:g}, nD+ {nDs[j]:.0e}, nD0 {nDs[k]:.0e}, '
                              f'nAr {nZs[m]:.0e}, nNe {nZs[n]:.0e}',
                        note='DREAM_avalanche reference data'))
    if int(os.environ.get('AVA_THESIS', 1)):
        ds, eec = thesis_avalanche_settings()
        pts.append(dict(kind='thesis', id='thesis', ds=ds, ref=None,
                        label=f'THESIS T = 8.35 eV, n_e = 1.58e20, Z_eff = 1.5 (ionised Ne), '
                              f'E = 35.5 V/m (E/Ec = {eec:.0f})',
                        note='analytic Hesslow rate (report only)'))
    return pts


def avalanche_settings(p):
    if p['kind'] == 'thesis':
        return copy.deepcopy(p['ds'])
    return DREAM_avalanche.gensettings(**p['args'])


def avalanche_case(p, nprocs):
    ds = avalanche_settings(p)
    tag = f'ava_{p["id"]}_np{nprocs}'
    out = run_dreami(ds, tag, nprocs, PETSC_FULL)
    if out is None:
        return None
    do = DREAMOutput(out)
    f = np.array(do.eqsys.f_hot.data)
    G = do.other.fluid.runawayRate[-1, 0] / do.eqsys.n_re[-1, 0]
    do.close()
    return f, G, True


def hesslow_rate(p):
    """Analytic growth rate (DREAM's Hesslow formula), from a fluid run on 1 rank."""
    ds = avalanche_settings(p)
    ds.hottailgrid.setEnabled(False)
    ds.timestep.setNt(2)
    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    tag = f'ava_{p["id"]}_hesslow'
    out = run_dreami(ds, tag, 1, '')
    if out is None:
        return None
    do = DREAMOutput(out)
    G = do.other.fluid.GammaAva[-1, 0]
    do.close()
    return G


# =====================================================================
# Driver
# =====================================================================
def check_family(name, points, case, tol, what):
    print(f'\n=== {name} ' + '=' * (68 - len(name)))
    success = True
    for p in points:
        print(f'{p["label"]}')
        if p.get('ref') is not None:
            print(f'    reference {p["ref"]:.4e}   ({p["note"]})')
        if DRY_RUN:
            for P in NPROCS:
                case(p, P)                       # writes settings only
            continue

        res1 = case(p, 1)
        if res1 is None:
            success = False
            continue
        f1, v1, resolved = res1

        # 1. one rank vs (transferred) reference
        ref = p.get('ref')
        if name == 'avalanche' and p.get('kind') == 'thesis':
            ref = hesslow_rate(p)
            tol_here, hard = AVA_ANALYTIC_TOL, False
        else:
            tol_here, hard = tol, True
        if not resolved:
            print(f'    REF     np  1   {what} not resolved (change per step < {MINREL:.0e}), skipped')
        elif ref is None or ref == 0:
            print(f'    REF     np  1   {what} {v1:.4e}   (no reference)')
        else:
            Delta = abs(v1 / ref - 1)
            ok = Delta <= tol_here
            if hard:
                success &= ok
            tag = ('ok' if ok else 'FAIL') if hard else ('ok' if ok else 'WARN')
            print(f'    REF     np  1   {what} {v1:.4e}   ref {ref:.4e}   '
                  f'Delta {Delta*100:7.3f} % (tol {tol_here*100:g} %)   {tag}')

        # 2. P ranks vs 1 rank
        for P in NPROCS[1:]:
            res = case(p, P)
            if res is None:
                success = False
                continue
            df = np.max(np.abs(res[0] - f1)) / np.max(np.abs(f1))
            dv = abs(res[1] - v1) / abs(v1) if v1 != 0 else abs(res[1])
            ok = df <= FTOL and (dv <= RTOL or not resolved)
            success &= ok
            print(f'    MPI     np {P:2d}   f_hot {df:.2e}   {what} {res[1]:.4e} ({dv:.2e})   '
                  f'{"ok" if ok else "FAIL"}')
    return success


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'dreami     {DREAMI}')
    print(f'ranks      {NPROCS}   ({" ".join(MPIRUN)} <n>)')
    print(f'post-TQ    T = {T_POST} eV (conductivity), {T_DREICER} eV (Dreicer), n = {N_POST:.2e}')
    print(f'PETSc      f_hot-only: "{PETSC_KIN}"   avalanche: "{PETSC_FULL}"')
    print(f'output     {OUTDIR}' + ('   [DRY RUN: settings only]' if DRY_RUN else ''))

    success = True
    if 'conductivity' in FAMILIES:
        success &= check_family('conductivity', conductivity_points(), conductivity_case,
                                COND_TOL, 'sigma')
    if 'dreicer' in FAMILIES:
        success &= check_family('dreicer', dreicer_points(), dreicer_case, DREI_TOL, 'rate')
    if 'avalanche' in FAMILIES:
        success &= check_family('avalanche', avalanche_points(), avalanche_case, AVA_TOL, 'Gamma')

    if DRY_RUN:
        print('\nDRY RUN: settings written, nothing run.')
        return 0
    print('\nALL CHECKS PASSED' if success else '\nSOME CHECKS FAILED')
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
