#!/usr/bin/env python3
#
# Kinetic-only DREAM run on a single momentum grid, with parameters
# matched to
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
# DREAM reduces to the same operator when the geometry is cylindrical, the
# background is prescribed and radial transport is disabled: the bounce
# average becomes the identity, the metric collapses to sqrt(g) = p^2 J,
# and only the radial term of Eq. (1) remains as a difference -- which is
# removed here by taking Nr = 1 with no transport.
#
# Unlike basic_kinetic_only.py, which splits the momentum range between
# the hot-tail and the runaway grid, the hot-tail grid is disabled here and
# the runaway grid alone covers the paper's domain [p_min, p_max]. There is
# then a single distribution, f_re, a single initial condition and no
# interface between grids.
#
# All quantities the paper fixes are listed under "Paper parameters" below
# and converted to DREAM's units in "Derived quantities", which are printed
# so that the correspondence can be checked.
#
# Run as
#
#   $ ./basic_f_re_only.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_f_re_only.h5
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
# Grid and time stepping
# =====================================================================
# The paper uses a base mesh of 48 x 8 with up to 6-7 further levels of
# refinement, concentrated where f varies most steeply. Near p_min the
# Maxwellian tail falls off on the scale v_t^2 / (2 p) ~ 0.02, far below
# what a uniform grid over [p_min, p_max] can afford, so the grid is
# bi-uniform in p instead: NP_SEP of the Np cells lie in [P_MIN, P_SEP]
# and the rest in [P_SEP, P_MAX]. The pitch grid is uniform.
Np     = 3000
NP_SEP = 1500               # cells in [P_MIN, P_SEP]
P_SEP  = 1.0
Nxi    = 100
Nr     = 1

# Time, in units of tau_c. The paper runs to T_final = 1 for its physics
# studies with an averaged time step near 0.004 tau_c; a shorter interval
# suffices for solver measurements.
T_FINAL_NORM = 1.0
Nt = 100

tMax = T_FINAL_NORM * tau_c

# =====================================================================
# Initial condition
# =====================================================================
# The paper's f0 of Sec. 6.7: a Maxwellian, uniform in xi, plus a small
# perturbation in the tail centred on (p, xi) = (40, -0.9),
#
#   f0 = 1/(v_t^3 pi^{3/2}) exp( (1 - sqrt(1+p^2)) / (v_t^2/2) )
#      + 1e-15 exp( -(p-40)^2 / 25 ) exp( -(xi+0.9)^2 / 0.0025 )
#
# The paper prints the exponent of the p factor without the minus sign,
# which would grow without bound away from p = 40; the decaying form is
# what "centred around (40, -0.9)" describes and is used here.
#
# The paper's f0 is normalised to unit density, whereas DREAM's f carries
# the density, so both terms are multiplied by n.
#
# f0 is tabulated on an input grid covering [P_MIN, P_MAX], which DREAM
# interpolates (linearly) onto its own grid. The input grid is independent
# of Np and Nxi, so it must itself resolve f0: the Maxwellian tail varies
# on the scale v_t^2 / (2 p) in p, hence the fine spacing below P_SEP, and
# the perturbation has widths 5 in p and 0.05 in xi.
BUMP_AMPLITUDE = 1e-15
BUMP_P,  BUMP_P_WIDTH2  = 40, 25.0      # the paper centres it at p = 40
BUMP_XI, BUMP_XI_WIDTH2 = +0.9, 0.0025

def f0(p, xi):
    """f0 on the grid (p, xi), with shape (1, nxi, np)."""
    P, XI = np.meshgrid(p, xi)
    maxwellian = np.exp((1 - np.sqrt(1 + P**2)) / (VHAT_T**2 / 2)) / (VHAT_T**3 * np.pi**1.5)
    bump = BUMP_AMPLITUDE * np.exp(-(P - BUMP_P)**2 / BUMP_P_WIDTH2) \
                          * np.exp(-(XI - BUMP_XI)**2 / BUMP_XI_WIDTH2)
    return (n * (maxwellian + bump))[np.newaxis, :, :]

NP_INIT_FINE, NP_INIT_COARSE, NXI_INIT = 2000, 2000, 400

p_init  = np.concatenate((np.linspace(P_MIN, P_SEP, NP_INIT_FINE + 1),
                          np.linspace(P_SEP, P_MAX, NP_INIT_COARSE + 1)[1:]))
xi_init = np.linspace(-1, 1, NXI_INIT + 1)
f_init  = f0(p_init, xi_init)

# =====================================================================
# Report
# =====================================================================
print('=' * 62)
print('Parameters matched to arXiv:2303.17019')
print('=' * 62)
print(f'  v_t (normalised)       {VHAT_T:.4f}')
print(f'  T_cold                 {T:.1f} eV   ({T/1e3:.3f} keV)')
print(f'  n_cold                 {n:.3e} m^-3')
print(f'  ln Lambda              {lnL:.2f}')
print(f'  tau_c                  {tau_c:.4e} s')
print(f'  E_c                    {E_c:.4e} V/m')
print(f'  E / E_c                {E_NORM:.2f}            (paper: 0.5 - 20)')
print(f'  E                      {E:.4e} V/m')
print(f'  B_0                    {B0:.3f} T')
print(f'  tau_s                  {tau_s:.4e} s')
print(f'  alpha = tau_c/tau_s    {alpha:.4f}          (paper: 0.001 - 0.3)')
print(f'  p range                [{P_MIN:.2f}, {P_MAX:.1f}]  (paper: [0.30, 60.0])')
print(f'    runaway grid         Np={Np}, Nxi={Nxi}')
print(f'      [{P_MIN:.2f}, {P_SEP:.1f}]         {NP_SEP} cells, dp = {(P_SEP-P_MIN)/NP_SEP:.2e}')
print(f'      [{P_SEP:.1f}, {P_MAX:.1f}]         {Np-NP_SEP} cells, dp = {(P_MAX-P_SEP)/(Np-NP_SEP):.2e}')
print(f'  Nr                     {Nr}')
print(f'  t_max                  {tMax:.4e} s   ({T_FINAL_NORM:.3f} tau_c)')
print('=' * 62)

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
ds.eqsys.E_field.setPrescribedData(E)
ds.eqsys.T_cold.setPrescribedData(T)
ds.eqsys.n_cold.setPrescribedData(n)
ds.eqsys.n_i.addIon(name='D', Z=Z, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n)

# Knock-on source: off, matching the solver studies of Sec. 6.3, where
# "the Fokker-Planck collision is turned on as usual while the knock-on
# source is turned off."
ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

# ---------------------------------------------------------------------
# Kinetic grid
# ---------------------------------------------------------------------
# The hot-tail grid is disabled; the runaway grid covers [P_MIN, P_MAX].
# Its lower limit is only taken from the settings when it is the only
# kinetic grid, as it is here.
ds.hottailgrid.setEnabled(False)

ds.runawaygrid.setEnabled(True)
ds.runawaygrid.setNp(Np)
ds.runawaygrid.setNxi(Nxi)
ds.runawaygrid.setPmin(P_MIN)
ds.runawaygrid.setPmax(P_MAX)
ds.runawaygrid.setBiuniformGrid(psep=P_SEP, npsep=NP_SEP)

ds.eqsys.f_re.setInitialValue(f_init, r=[0], p=p_init, xi=xi_init)
ds.eqsys.f_re.setAdvectionInterpolationMethod(DistFunc.AD_INTERP_UPWIND)

# Upper boundary, p = P_MAX: the flux out of the grid is extrapolated from
# the interior. (The lower boundary is discussed at the end of this file.)
ds.eqsys.f_re.setBoundaryCondition(DistFunc.BC_PHI_CONST)

# Synchrotron damping, the alpha R(f) term of Eq. (4); its strength
# follows from B_0 above.
ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)

# ---------------------------------------------------------------------
# Geometry: cylindrical, one radial cell, no transport
# ---------------------------------------------------------------------
# A single radial cell removes the radial term that distinguishes DREAM's
# equation from the paper's. Cylindrical geometry additionally gives
# B_min = B_max, hence xi_T = 0 and no trapped particles -- so the bounce
# average reduces to the identity and the metric to sqrt(g) = p^2 J, which
# is the paper's Jacobian.
ds.radialgrid.setB0(B0)
ds.radialgrid.setMinorRadius(0.22)
ds.radialgrid.setWallRadius(0.22)
ds.radialgrid.setNr(Nr)

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
ds.output.setFilename('output_f_re_only.h5')

ds.save(os.path.join(OUT_DIR, 'dream_settings_f_re_only.h5'))

with open(os.path.join(OUT_DIR, 'petsc_num_timesteps.txt'), 'w') as f:
    f.write(str(int(Nt)) + '\n')

# =====================================================================
# Remaining differences from the paper
# =====================================================================
print()
print('Differences that could not be removed from the settings:')
print()
print('  p_min   The domain starts at p_min = 3 v_t, as in the paper, but')
print('          the paper imposes a Dirichlet Maxwellian there, whereas')
print('          DREAM has no boundary condition at the lower edge of a')
print('          runaway grid that is the only kinetic grid: the flux')
print('          through p = p_min is zero. Particles slowing down below')
print('          the critical momentum therefore pile up at p_min instead')
print('          of leaving the domain.')
print()
print('  p_max   The paper applies a Neumann condition at p_max. DREAM')
print('          extrapolates the flux out of the grid from the interior.')
print()
print('  mesh    The paper adapts the mesh, reaching 6-8 further levels of')
print('          refinement over a 48 x 8 base. The grid here is fixed,')
print('          refined in p below P_SEP only, so iteration counts are')
print('          not compared at equal resolution.')
print()
print('  moments DREAM retains the fluid unknowns as equations even when')
print('          their values are prescribed, so the moment operators are')
print('          still assembled. They no longer feed back into f, but the')
print('          dense rows they produce remain in the system unless the')
print('          corresponding blocks are dropped from the matrix.')
print()
print('Reference points from the paper, for comparison:')
print('  GMRES iterations/solve   ~4.5 (Tables 4, 5), ~6 at 4.4M cells')
print('  preconditioner           BoomerAMG, Euclid as the smoother')
print()