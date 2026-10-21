#!/usr/bin/env python3
#
# Kinetic-only DREAM run, with parameters matched to
#
#   J. Rudi, M. Heldman, E. M. Constantinescu, Q. Tang, X.-Z. Tang,
#   "Scalable Implicit Solvers with Dynamic Mesh Adaptation for a
#    Relativistic Drift-Kinetic Fokker-Planck-Boltzmann Model",
#   arXiv:2303.17019.
#
# The purpose is a like-for-like comparison of the linear solver on the
# momentum-space operator. The paper solves
#
#   df/dt - E ( xi df/dp + (1-xi^2)/p df/dxi ) = C(f) + alpha R(f) + S(f)
#
# on [p_min, p_max] x [-1, 1] with every background quantity held fixed.
#
# RADIAL VARIANT of diff_grid_paper_kinetic.py. Same momentum-space setup,
# but with NR > 1 radial cells and Rechester-Rosenbluth radial transport
# (D_rr = pi q R0 (dB/B)^2 |v_par|) switched on for both f_hot and f_re,
# with f = 0 outside the plasma. This adds the radial term of DREAM's
# Eq. (1) that the paper does not have, and couples the Nr momentum-space
# blocks through a tridiagonal-in-r structure.
#
# Run as
#
#   $ ./diff_grid_paper_kinetic_radial.py                    # defaults
#   $ LEVEL=3 NR=16 DBB=1e-4 ./diff_grid_paper_kinetic_radial.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_paper_radial_L2_Nr8.h5
#
# ###################################################################

import os
import sys
import numpy as np

DREAM_ROOT = os.path.expanduser(os.environ.get('DREAM_ROOT', '~/Desktop/DREAM'))
OUT_DIR = os.path.expanduser(
    os.environ.get('KINETIC_OUT_DIR', os.path.join(DREAM_ROOT, 'build', 'iface')))

sys.path.append(os.path.join(DREAM_ROOT, 'py'))

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport

# =====================================================================
# Physical constants (SI)
# =====================================================================
e     = 1.602176634e-19     # elementary charge            [C]
me    = 9.1093837015e-31    # electron rest mass           [kg]
c     = 2.99792458e8        # speed of light               [m/s]
eps0  = 8.8541878128e-12    # vacuum permittivity          [F/m]
mc2   = me * c**2 / e       # electron rest energy         [eV]

# =====================================================================
# Paper parameters
# =====================================================================
# Normalised thermal velocity. The paper's initial condition (Sec. 6.6)
#
#   f0 = 1/(v_t^3 pi^{3/2}) exp( (1 - sqrt(1+p^2)) / (v_t^2/2) )
#
# reduces non-relativistically to exp(-p^2/v_t^2), hence v_t = sqrt(2T/mc^2).
VHAT_T = 0.1

# Normalised electric field, in units of the Connor-Hastie critical field
# E_c := m_e c / (e tau_c)  (Sec. 2). The paper's studies cover [0.5, 20];
# E = 2 is stated to be above the avalanche threshold, and E = 5 is used
# for the scalability runs.
E_NORM = 5.0

# Damping intensity alpha := tau_c / tau_s (Sec. 2.2), practical range
# 0.001 to 0.3. Rather than prescribing alpha directly -- DREAM derives it
# from the magnetic field -- B is chosen below and the resulting alpha is
# reported, so that it can be checked against this target.
ALPHA_TARGET = 0.1

# Momentum domain (Sec. 5.2). p_min = 3 v_t is where the paper imposes a
# Dirichlet Maxwellian; p_max = 60.
P_MIN = 3 * VHAT_T
P_MAX = 60.0

# Charge number: a single species, fully ionised.
Z = 1

# Background electron density. The paper gives this only through tau_c;
# any value consistent with the normalisation will do, so a typical
# post-disruption density is used and E_c computed from it.
n = 5e19                    # [m^-3]

# Hot/runaway interface. The momentum range is covered by two grids rather
# than the paper's one, joined here by the conservative flux-matching
# interface of Eqs. (67)-(72). This is a structural difference from the
# paper and shows up in the matrix as the sparse off-diagonal blocks
# J_hr, J_rh confined to p = P_RE.
P_RE = 1.0

# =====================================================================
# Derived quantities
# =====================================================================
T = 0.5 * VHAT_T**2 * mc2   # cold temperature [eV], from v_t = sqrt(2T/mc^2)

# Coulomb logarithm, DREAM Eq. (18) in the Hoppe et al. paper:
#   ln L_0 = 14.9 + ln(T/1keV) - 0.5 ln(n/1e20 m^-3)
lnL = 14.9 + np.log(T / 1e3) - 0.5 * np.log(n / 1e20)

# Relativistic collision time, paper Eq. (2).
tau_c = 4 * np.pi * eps0**2 * me**2 * c**3 / (e**4 * n * lnL)

# Connor-Hastie critical field, E_c := m_e c / (e tau_c).
E_c = me * c / (e * tau_c)

# Synchrotron time scale, tau_s := 6 pi eps0 m_e^3 c^3 / (e^4 B^2), so that
# alpha = tau_c/tau_s = ALPHA_TARGET fixes B.
B0 = np.sqrt(ALPHA_TARGET * 6 * np.pi * eps0 * me**3 * c**3 / (e**4 * tau_c))
tau_s = 6 * np.pi * eps0 * me**3 * c**3 / (e**4 * B0**2)
alpha = tau_c / tau_s

# Electric field in DREAM's units.
E = E_NORM * E_c

# =====================================================================
# Grids: DIFFERENT resolution on the two kinetic grids
# =====================================================================
# This is the point of maintaining two grids rather than one. The two
# populations require opposite resolutions:
#
#            hot-tail  [0, P_RE]        runaway  [P_RE, P_MAX]
#   shape    steep in p, near-isotropic smooth in p, narrow beam in xi
#   needs    MANY p, FEW xi             FEW p, MANY xi (clustered)
#
# A single tensor grid would cost Np_fine * Nxi_fine; two grids cost
# Np_fine*Nxi_coarse + Np_coarse*Nxi_fine, turning a product into a sum.
# This is the same requirement the paper meets by adapting one mesh.
#
# The numbers below come from resolving the two physical scales:
#
#   HOT, p   The Maxwellian exp[(1-gamma)/(vt^2/2)] e-folds over
#            ~ vt^2/(2p) = 0.017 at p = 0.3, so dp <~ 4e-3, Np >~ 250.
#
#   HOT, xi  Nearly isotropic: nu_D ~ p^-3 is enormous at thermal momenta
#            and isotropises faster than E can focus. The anisotropy scale
#            is 1-|xi| ~ 1/(E p) ~ 0.2 at p ~ 1, so dxi <~ 0.07, Nxi >~ 30.
#
#   RE, p    Smooth tail; the perturbation has width sqrt(25) = 5 in p, so
#            ~25 points across it gives dp ~ 0.2, Np >~ 300.
#
#   RE, xi   THE BINDING CONSTRAINT. The beam width follows from the
#            balance between pitch scattering (rate ~ 2/p^2) and E-field
#            focusing (rate ~ 2E/p):
#
#                1 - |xi| ~ 1/(E p)
#
#            which is 5e-3 at p = 40 and 3.3e-3 at p = 60. Three points
#            inside the beam on a UNIFORM grid needs Nxi >~ 1200-1800.
#            With BIUNIFORM clustering near xi = +-1 (below) the same
#            resolution is reached with roughly 8x fewer points.
#
# LEVEL refines all four numbers by 2^LEVEL, so each level is 4x the
# unknowns. Level 2 is the coarsest that resolves everything above.
LEVEL = int(os.environ.get('LEVEL', 2))
_f = 2 ** LEVEL

NP_HOT  = 125 * _f          # many p
NXI_HOT =  16 * _f          # few xi

NP_RE   =  75 * _f          # few p
NXI_RE  = 100 * _f          # many xi

# =====================================================================
# Radial grid and transport
# =====================================================================
# Number of radial cells. Each cell carries a full copy of both momentum
# grids, so the matrix size scales linearly with NR.
Nr = int(os.environ.get('NR', 8))

# Minor radius (= wall radius) [m].
A_MINOR = 0.22

# Magnetic perturbation level dB/B for Rechester-Rosenbluth transport,
#
#   D_rr = pi q R0 (dB/B)^2 |v_par|,
#
# with R0 -> 1 m in cylindrical geometry (DREAM replaces R0 = inf by 1).
# For relativistic electrons D_rr ~ pi c (dB/B)^2, so the radial
# diffusion time is a^2 / D_rr ~ 0.12 tau_c at dB/B = 1e-4: comparable to
# the simulated interval, i.e. transport visibly shapes the solution
# without completely draining the plasma. DBB=0 switches transport off.
DBB = float(os.environ.get('DBB', 1e-4))
D_RR_REL = np.pi * c * DBB**2       # [m^2/s], v_par = c, q = R0 = 1

# Cluster the runaway pitch grid near xi = +-1, the static analogue of the
# paper's AMR. thetasep is the pitch angle separating the two uniform
# regions and nthetasep_frac the fraction of points placed below it.
#
# NOTE: check the sign convention against your own run before trusting
# this. DREAM's theta grid runs [0, pi] <-> xi = [+1, -1], so thetasep
# clusters toward xi = +1; in your earlier runs the beam grew at xi = +1
# (the blue cut), consistent with that, but the paper states xi = -1 for
# E > 0. If the beam in your output sits at xi = -1, raise thetasep
# toward pi instead.
BIUNIFORM_RE = True
THETASEP     = 0.5
NTHETASEP_FRAC = 0.75

# Advection interpolation. UPWIND is monotone and keeps the operator an
# M-matrix, but is first order: D_num = |A| dp / 2 ~ 0.1 in the tail
# against a physical C_A ~ 5e-3. QUICK is the paper's scalability-run
# scheme (Sec. 6.4), second order, and still LINEAR -- it never calls
# GetFluxLimiterR -- so LINEAR_IMPLICIT remains valid. It is not monotone;
# watch for negative f.
ADV_INTERP = {
    'upwind': DistFunc.AD_INTERP_UPWIND,
    'quick':  DistFunc.AD_INTERP_QUICK,
}[os.environ.get('ADV', 'upwind').lower()]

# =====================================================================
# Time stepping
# =====================================================================
# Time in units of tau_c. The paper runs to T_final = 1 for its physics
# studies with an averaged time step near 0.004 tau_c; a shorter interval
# suffices for solver measurements. Raise T_FINAL_NORM to 1.0 to see the
# tail perturbation actually evolve.
#
# dt is held FIXED across LEVEL. As h -> 0 the operator M = I/dt + L
# becomes increasingly dominated by L, with kappa ~ h^-2, which is the
# honest test of a preconditioner. (The paper instead shrinks dt with
# refinement: 1e-5 at 1.4M cells, 1e-6 at 4.4M -- a milder test.)
T_FINAL_NORM = 0.5
Nt   = 500
tMax = T_FINAL_NORM * tau_c

# =====================================================================
# Initial condition
# =====================================================================
# The paper's f0 of Sec. 6.7: a Maxwellian, uniform in xi, plus a small
# perturbation in the tail centred on (p, xi) = (40, -0.9),
#
#   f0 = 1/(vt^3 pi^{3/2}) exp( (1 - sqrt(1+p^2)) / (vt^2/2) )
#      + 1e-15 exp( -(p-40)^2 / 25 ) exp( -(xi+0.9)^2 / 0.0025 )
#
# The paper prints the exponent of the p factor without the minus sign,
# which would grow without bound away from p = 40; the decaying form is
# what "centred around (40, -0.9)" describes and is used here.
#
# The paper's f0 is normalised to unit density, whereas DREAM's f carries
# the density, so both terms are multiplied by n.
#
# BUMP_XI: the paper puts the perturbation at xi = -0.9, which for their
# sign convention is the ACCELERATING side. If in your run the field
# accelerates toward xi = +1, a bump at -0.9 is decelerating and will not
# run away. Flip the sign here if the bump drifts to lower p.
BUMP_AMPLITUDE = 1e-15
BUMP_P,  BUMP_P_WIDTH2  = 40.0, 25.0
BUMP_XI, BUMP_XI_WIDTH2 = -0.9, 0.0025


# The initial f is taken uniform in r: the background (n, T, E) is
# prescribed uniform as well, so the only radial gradient is the one
# created by the f = 0 boundary condition at r = a, which the transport
# then propagates inwards.
r_init = np.linspace(0.0, A_MINOR, 11)


def f0(p, xi):
    """f0 tabulated on (r, p, xi), returned with shape (nr, nxi, np)."""
    P, XI = np.meshgrid(p, xi)
    maxwellian = np.exp((1 - np.sqrt(1 + P**2)) / (VHAT_T**2 / 2)) \
                 / (VHAT_T**3 * np.pi**1.5)
    bump = BUMP_AMPLITUDE * np.exp(-(P - BUMP_P)**2 / BUMP_P_WIDTH2) \
                          * np.exp(-(XI - BUMP_XI)**2 / BUMP_XI_WIDTH2)
    f_pxi = n * (maxwellian + bump)
    return np.repeat(f_pxi[np.newaxis, :, :], r_init.size, axis=0)


# Two separate tables, one per grid, each sampled to resolve what lives on
# that grid. The input tables are independent of NP_*/NXI_* -- DREAM
# interpolates them onto whatever grid it builds -- so they must resolve
# f0 in their own right, regardless of LEVEL.
#
#   hot table: fine in p (Maxwellian varies on the scale vt), coarse in xi
#              (f0 is uniform in xi there).
#   re table:  moderate in p, fine in xi (the bump has width sqrt(0.0025)
#              = 0.05, so ~20 points across it needs dxi ~ 2.5e-3).
p_hot_init  = np.linspace(0.0,   P_RE,  2001)
xi_hot_init = np.linspace(-1.0,  1.0,     65)

p_re_init   = np.linspace(P_RE,  P_MAX, 1001)
xi_re_init  = np.linspace(-1.0,  1.0,    801)

f_hot_init = f0(p_hot_init, xi_hot_init)
f_re_init  = f0(p_re_init,  xi_re_init)

# =====================================================================
# Report
# =====================================================================
N_hot = NP_HOT * NXI_HOT * Nr
N_re  = NP_RE  * NXI_RE  * Nr
N_tot = N_hot + N_re

# Beam width 1-|xi| ~ 1/(E p) at the top of the runaway grid, against the
# pitch spacing actually delivered (uniform estimate).
delta_beam = 1.0 / (E_NORM * P_MAX)
dxi_re     = 2.0 / NXI_RE

print('=' * 70)
print(f'Parameters matched to arXiv:2303.17019          [LEVEL = {LEVEL}]')
print('=' * 70)
print(f'  v_t (normalised)       {VHAT_T:.4f}')
print(f'  T_cold                 {T:.1f} eV   ({T/1e3:.3f} keV)')
print(f'  n_cold                 {n:.3e} m^-3')
print(f'  ln Lambda              {lnL:.2f}')
print(f'  tau_c                  {tau_c:.4e} s')
print(f'  E_c                    {E_c:.4e} V/m')
print(f'  E / E_c                {E_NORM:.2f}            (paper: 0.5 - 20)')
print(f'  E                      {E:.4e} V/m')
print(f'  B_0                    {B0:.3f} T')
print(f'  alpha = tau_c/tau_s    {alpha:.4f}          (paper: 0.001 - 0.3)')
print(f'  t_max                  {tMax:.4e} s   ({T_FINAL_NORM:.3f} tau_c)')
print(f'  dt                     {tMax/Nt:.4e} s   (fixed across LEVEL)')
print(f'  advection scheme       {os.environ.get("ADV", "upwind")}')
print('-' * 70)
print(f'  hot-tail grid   p in [0, {P_RE:.1f}]      '
      f'Np = {NP_HOT:6d}   Nxi = {NXI_HOT:5d}   -> {N_hot:9,d}')
print(f'    dp  = {P_RE/NP_HOT:.2e}   (need <~ 4e-3 for the bulk)')
print(f'    dxi = {2.0/NXI_HOT:.2e}   (need <~ 7e-2, near-isotropic)')
print(f'  runaway grid    p in [{P_RE:.1f}, {P_MAX:.0f}]    '
      f'Np = {NP_RE:6d}   Nxi = {NXI_RE:5d}   -> {N_re:9,d}')
print(f'    dp  = {(P_MAX-P_RE)/NP_RE:.2e}   (need <~ 2e-1 for the bump)')
print(f'    dxi = {dxi_re:.2e}   (beam 1-|xi| ~ {delta_beam:.2e} at p = {P_MAX:.0f})')
if dxi_re > delta_beam / 3 and not BIUNIFORM_RE:
    print(f'    WARNING: uniform pitch grid does not resolve the beam.')
    print(f'             Need Nxi >~ {int(np.ceil(6.0/delta_beam)):d}, '
          f'or enable BIUNIFORM_RE.')
elif BIUNIFORM_RE:
    print(f'    biuniform: thetasep = {THETASEP}, '
          f'frac = {NTHETASEP_FRAC}  (clustered near xi = +-1)')
print('-' * 70)
print(f'  Nr                     {Nr}   (a = {A_MINOR} m, dr = {A_MINOR/Nr:.3e} m)')
if DBB > 0:
    print(f'  radial transport       Rechester-Rosenbluth, dB/B = {DBB:.1e}, BC f = 0')
    print(f'    D_rr (v = c)         {D_RR_REL:.3e} m^2/s')
    print(f'    a^2 / D_rr           {A_MINOR**2/D_RR_REL/tau_c:.3e} tau_c')
else:
    print(f'  radial transport       OFF (DBB = 0)')
print(f'  matrix size            {N_tot:,d}')
for P in (1, 8, 32, 256):
    print(f'    on {P:4d} rank(s): {N_tot//P:10,d} unknowns/rank'
          + ('   <- paper regime' if 100 <= N_tot//P <= 2000 else ''))
print('=' * 70)

# =====================================================================
# Settings
# =====================================================================
ds = DREAMSettings()

# Partially screened collision frequencies, as the paper uses when the
# screening effect is included (Sec. 2.1). With Z = 1 and full ionisation
# the screening correction g_i(p) vanishes, so this matches their bare-Z
# treatment.
ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

# ---------------------------------------------------------------------
# Fixed background
# ---------------------------------------------------------------------
# Everything the paper holds constant: "In the current study all those
# values are given as constant based on practical devices such as ITER."
#
# Because n_cold, T_cold and n_i are all prescribed constants, the
# collision coefficients are rebuilt once (CollisionQuantity::Rebuild is
# guarded by parametersHaveChanged()) and M is the SAME MATRIX at every
# timestep -- only S changes. Worth exploiting with
# KSPSetReusePreconditioner once the solver is settled.
ds.eqsys.E_field.setPrescribedData(E)
ds.eqsys.T_cold.setPrescribedData(T)
ds.eqsys.n_cold.setPrescribedData(n)
ds.eqsys.n_i.addIon(name='D', Z=Z, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n)

# Knock-on source: off, matching the solver studies of Sec. 6.3, where
# "the Fokker-Planck collision is turned on as usual while the knock-on
# source is turned off."
ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

# ---------------------------------------------------------------------
# Hot-tail grid: [0, P_RE], many p, few xi
# ---------------------------------------------------------------------
ds.hottailgrid.setNp(NP_HOT)
ds.hottailgrid.setNxi(NXI_HOT)
ds.hottailgrid.setPmax(P_RE)

ds.eqsys.f_hot.setInitialValue(f_hot_init, r=r_init, p=p_hot_init, xi=xi_hot_init)
ds.eqsys.f_hot.setBoundaryCondition(DistFunc.BC_F_0)
ds.eqsys.f_hot.setAdvectionInterpolationMethod(ADV_INTERP)

# Synchrotron damping, the alpha R(f) term of Eq. (4); strength follows
# from B_0 above.
ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)

# ---------------------------------------------------------------------
# Runaway grid: [P_RE, P_MAX], few p, many xi, clustered
# ---------------------------------------------------------------------
# The same equation is solved on both grids, but nu_s, nu_D ~ p^-3, so the
# operator is diffusion-dominated on the hot grid and close to pure
# advection here. That difference is the reason the fieldsplit
# preconditioner treats the two blocks separately (AMG on f_hot, ILU or
# AIR on f_re).
ds.runawaygrid.setEnabled(True)
ds.runawaygrid.setNp(NP_RE)
ds.runawaygrid.setNxi(NXI_RE)
ds.runawaygrid.setPmax(P_MAX)

if BIUNIFORM_RE:
    ds.runawaygrid.setBiuniformGrid(thetasep=THETASEP,
                                    nthetasep_frac=NTHETASEP_FRAC)

ds.eqsys.f_re.setInitialValue(f_re_init, r=r_init, p=p_re_init, xi=xi_re_init)
ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
ds.eqsys.f_re.setAdvectionInterpolationMethod(ADV_INTERP)

# ---------------------------------------------------------------------
# Geometry: cylindrical, Nr radial cells
# ---------------------------------------------------------------------
# Cylindrical geometry gives B_min = B_max, hence xi_T = 0 and no trapped
# particles -- so the bounce average reduces to the identity and the
# momentum-space metric to sqrt(g) = p^2 J, the paper's Jacobian. Unlike
# the Nr = 1 script, the radial term is now kept.
ds.radialgrid.setB0(B0)
ds.radialgrid.setMinorRadius(A_MINOR)
ds.radialgrid.setWallRadius(A_MINOR)
ds.radialgrid.setNr(Nr)

# ---------------------------------------------------------------------
# Radial transport: Rechester-Rosenbluth on both kinetic grids
# ---------------------------------------------------------------------
# Applied to f_hot and f_re alike, so that electrons crossing p = P_RE see
# the same radial diffusion on either side of the interface. BC_F_0 sets
# f = 0 outside the plasma, i.e. particles reaching r = a are lost.
if DBB > 0:
    ds.eqsys.f_hot.transport.setMagneticPerturbation(DBB)
    ds.eqsys.f_hot.transport.setBoundaryCondition(Transport.BC_F_0)
    ds.eqsys.f_re.transport.setMagneticPerturbation(DBB)
    ds.eqsys.f_re.transport.setBoundaryCondition(Transport.BC_F_0)

# ---------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.solver.preconditioner.setEnabled(False)
ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_AMG)

ds.other.include('fluid', 'nu_s', 'nu_D')

ds.timestep.setTmax(tMax)
ds.timestep.setNt(Nt)

ds.output.setTiming(stdout=True, file=True)
TAG = f'paper_radial_L{LEVEL}_Nr{Nr}'
ds.output.setFilename(f'output_{TAG}.h5')

settings_file = os.path.join(OUT_DIR, f'dream_settings_{TAG}.h5')
ds.save(settings_file)

with open(os.path.join(OUT_DIR, 'petsc_num_timesteps.txt'), 'w') as fh:
    fh.write(str(int(Nt)) + '\n')

print()
print(f'Wrote {settings_file}')
print(f'Run:  cd {OUT_DIR} && ./dreami dream_settings_{TAG}.h5')

# =====================================================================
# Remaining differences from the paper
# =====================================================================
print()
print('Differences that could not be removed from the settings:')
print()
print('  radial  (this variant) Nr > 1 with Rechester-Rosenbluth')
print('          transport: the paper has no radial dimension. The')
print('          matrix becomes Nr momentum-space blocks coupled by a')
print('          tridiagonal-in-r diffusion term.')
print()
print('  p_min   The paper truncates at p_min = 3 v_t and imposes a')
print('          Dirichlet Maxwellian there, so the thermal bulk is never')
print('          resolved. DREAM\'s hot-tail grid begins at p = 0, so the')
print('          bulk is resolved kinetically. This is where nu_s, nu_D ~')
print('          p^-3 are largest and the operator stiffest, so it affects')
print('          both conditioning and cost. The nearest equivalent is the')
print('          superthermal collision mode, which takes T_cold -> 0 and')
print('          drains the bulk off the grid instead.')
print()
print('  mesh    The paper adapts the mesh dynamically, reaching 6-8 levels')
print('          of refinement over a 48 x 8 base. The two grids here are')
print('          static; the biuniform pitch grid on the runaway side is')
print('          the nearest static analogue.')
print()
print('  split   Two grids joined at p = P_RE, against the paper\'s single')
print('          adaptive mesh. The interface contributes the sparse')
print('          off-diagonal blocks J_hr, J_rh, which the paper has no')
print('          counterpart for.')
print()
print('  moments DREAM retains the fluid unknowns as equations even when')
print('          their values are prescribed, so the moment operators are')
print('          still assembled. They no longer feed back into f, but the')
print('          dense rows they produce remain in the system unless the')
print('          corresponding blocks are dropped from the matrix.')
print()
print('Reference points from the paper, for comparison:')
print('  GMRES iterations/solve   ~4.5 (Tables 4, 5), 6.0 at 2.9M and 4.4M')
print('  stopping criterion       rtol = 1e-6')
print('  preconditioner           BoomerAMG, Euclid as the smoother')
print('  scheme                   QUICK (Sec. 6.4), MUSCL (Sec. 6.2)')
print()