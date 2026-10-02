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
# DREAM reduces to the same operator when the geometry is cylindrical, the
# background is prescribed and radial transport is disabled: the bounce
# average becomes the identity, the metric collapses to sqrt(g) = p^2 J,
# and only the radial term of Eq. (1) remains as a difference -- which is
# removed here by taking Nr = 1 with no transport.
#
# All quantities the paper fixes are listed under "Paper parameters" below
# and converted to DREAM's units in "Derived quantities", which are printed
# so that the correspondence can be checked.
#
# Run as
#
#   $ ./basic_kinetic_paper.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_paper.h5
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
# refinement, so its effective resolution in pitch is far higher than the
# base suggests. A uniform grid is used here; Nxi is raised well above the
# value used in earlier runs, since the distribution becomes "extreme[ly]
# anisotropic ... increasingly aggravated for higher electron energies"
# (Sec. 1) and an under-resolved pitch direction would flatter the
# iteration counts.
Np   = 500
Nxi  = 64
Nr   = 1

# Time, in units of tau_c. The paper runs to T_final = 1 for its physics
# studies with an averaged time step near 0.004 tau_c; a shorter interval
# suffices for solver measurements.
T_FINAL_NORM = 0.05
Nt = 100

tMax = T_FINAL_NORM * tau_c

# The momentum range is covered by two grids rather than the paper's one,
# joined at p = P_RE by the conservative flux-matching interface. The two
# populations call for opposite resolutions -- the hot region varies
# steeply in p and is nearly isotropic in xi, the runaway tail the reverse
# -- which is the same requirement the paper meets by adapting a single
# mesh. The interface is therefore a structural difference from the paper
# and shows up in the matrix as the sparse off-diagonal blocks coupling
# the two kinetic unknowns at p = P_RE.
P_RE = 1.0                  # hot/runaway interface

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
print(f'  p range                [{P_MIN:.2f}, {P_MAX:.1f}]  (paper: single grid)')
print(f'    hot-tail grid        [0, {P_RE:.1f}]        Np={Np}, Nxi={Nxi}')
print(f'    runaway grid         [{P_RE:.1f}, {P_MAX:.1f}]      Np={Np}, Nxi={Nxi}')
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
# Hot-tail grid: [0, P_RE].
ds.hottailgrid.setNxi(Nxi)
ds.hottailgrid.setNp(Np)
ds.hottailgrid.setPmax(P_RE)

ds.eqsys.f_hot.setInitialProfiles(n0=n, T0=T)
ds.eqsys.f_hot.setBoundaryCondition(DistFunc.BC_F_0)
ds.eqsys.f_hot.setAdvectionInterpolationMethod(DistFunc.AD_INTERP_UPWIND)

# Synchrotron damping, the alpha R(f) term of Eq. (4). Enabled here,
# unlike in the earlier runs; its strength follows from B_0 above.
ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)

# Runaway grid: [P_RE, P_MAX]. The same equation is solved on both, but the
# collision frequencies scale as p^-3, so the operator is diffusion-
# dominated on the hot grid and close to pure advection here.
ds.runawaygrid.setEnabled(True)
ds.runawaygrid.setNp(Np)
ds.runawaygrid.setNxi(Nxi)
ds.runawaygrid.setPmax(P_MAX)
ds.eqsys.f_re.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_INCLUDE)
ds.eqsys.f_re.setAdvectionInterpolationMethod(DistFunc.AD_INTERP_UPWIND)

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
ds.output.setFilename('output_paper.h5')

ds.save(os.path.join(OUT_DIR, 'dream_settings_paper.h5'))

with open(os.path.join(OUT_DIR, 'petsc_num_timesteps.txt'), 'w') as f:
    f.write(str(int(Nt)) + '\n')

# =====================================================================
# Remaining differences from the paper
# =====================================================================
print()
print('Differences that could not be removed from the settings:')
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
print('  mesh    The paper adapts the mesh, reaching 6-8 further levels of')
print('          refinement over a 48 x 8 base. The grid here is uniform,')
print('          so iteration counts are not compared at equal resolution.')
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