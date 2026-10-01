#!/usr/bin/env python3
#
# Kinetic-only variant of basic.py, for linear-solver experiments.
#
# Two differences from basic.py:
#
#   1. Every unknown that DREAM lets us prescribe from the settings is fixed
#      to a physically consistent constant (see "Fixed quantities" below), so
#      that the only genuinely evolving unknowns are the distribution
#      functions f_hot and f_re. The remaining fluid/scalar unknowns
#      (n_hot, n_re, j_hot, j_re, j_ohm, j_tot, W_cold, psi_p, I_p, ...) are
#      moments of f or algebraic functions of it; DREAM always keeps them as
#      equations of the system and they cannot be removed from Python.
#
#   2. No radial transport and a single radial cell (Nr = 1), so the
#      kinetic block is a pure (p, xi) problem with no radial coupling.
#
# Run as
#
#   $ ./basic_kinetic_only.py
#   $ cd $DREAM_ROOT/build/iface && ./dreami dream_settings_kinetic.h5
#
# DREAM_ROOT (environment variable, default ~/Desktop/DREAM) is the DREAM
# tree that provides the DREAM python package and the build/iface directory.
# KINETIC_OUT_DIR (optional) overrides where the settings are written.
#
# The settings are written to dream_settings_kinetic.h5 (not dream_settings.h5),
# so the settings of other runs are not overwritten. dreami writes its
# petsc_mat_serial_step* / petsc_rhs_serial_step* files into the directory
# it is run from, so run it from a separate directory if you want to keep
# the files of an earlier run.
#
# ###################################################################

import os
import numpy as np
import sys

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

ds = DREAMSettings()
ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

# Physical parameters
E = 6       # Electric field strength (V/m)
n = 5e19    # Electron density (m^-3)
T = 100     # Temperature (eV)

# Grid parameters
pMax = 1    # maximum momentum in units of m_e*c
Np   = 500  # number of momentum grid points
Nr   = 1    # a single radial cell: no radial structure
Nxi  = 10   # number of pitch grid points
tMax = 1e-4 # simulation time in seconds
Nt   = 20   # number of time steps

# ---------------------------------------------------------------------
# Fixed quantities (everything that can be prescribed)
# ---------------------------------------------------------------------
# Electric field and cold-electron temperature: constant.
ds.eqsys.E_field.setPrescribedData(E)
ds.eqsys.T_cold.setPrescribedData(T)

# Ions: fully ionised deuterium at the electron density, so charge
# neutrality gives n_free = n.
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n)

# Cold-electron density: fixed to the free-electron density. This is the
# value the self-consistent equation (n_cold = n_free - n_hot - n_re)
# gives at t = 0, where the Maxwellian carries all the density and the
# runaway density is zero.
ds.eqsys.n_cold.setPrescribedData(n)

# Disable avalanche generation (no secondary runaway source in f_re)
ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)

# ---------------------------------------------------------------------
# Kinetic part: the only evolving unknowns
# ---------------------------------------------------------------------
# Hot-tail grid
ds.hottailgrid.setNxi(Nxi)
ds.hottailgrid.setNp(Np)
ds.hottailgrid.setPmax(pMax)

# Initial hot electron Maxwellian. No radial transport is set for f_hot
# (basic.py sets a magnetic perturbation and a transport BC here).
ds.eqsys.f_hot.setInitialProfiles(n0=n, T0=T)

# Boundary condition at pMax
ds.eqsys.f_hot.setBoundaryCondition(DistFunc.BC_F_0) # F=0 outside the boundary
ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)
ds.eqsys.f_hot.setAdvectionInterpolationMethod(DistFunc.AD_INTERP_UPWIND)

# Runaway grid
ds.runawaygrid.setEnabled(True)
ds.runawaygrid.setNp(Np)
ds.runawaygrid.setNxi(Nxi)
ds.runawaygrid.setBiuniformGrid(thetasep=0.5, nthetasep_frac=0.75)
ds.runawaygrid.setPmax(30)

# Radial grid: one cell
ds.radialgrid.setB0(5)
ds.radialgrid.setMinorRadius(0.22)
ds.radialgrid.setWallRadius(0.22)
ds.radialgrid.setNr(Nr)

# Solver
ds.solver.setType(Solver.LINEAR_IMPLICIT) # semi-implicit time stepping
ds.solver.preconditioner.setEnabled(False)
ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_ILU)
#ds.solver.setLinearSolver(Solver.LINEAR_SOLVER_AMG)

# Other quantities to save to output
ds.other.include('fluid','nu_s','nu_D')

# Time stepper
ds.timestep.setTmax(tMax)
ds.timestep.setNt(Nt)

ds.output.setTiming(stdout=True, file=True)
ds.output.setFilename('output_kinetic.h5')

# Save settings
ds.save(os.path.join(OUT_DIR, 'dream_settings_kinetic.h5'))

# Number of time steps, for the parallel matrix-replay path in
# iface/Main.cpp (see basic.py).
with open(os.path.join(OUT_DIR, 'petsc_num_timesteps.txt'), 'w') as f:
    f.write(str(int(Nt)) + '\n')
