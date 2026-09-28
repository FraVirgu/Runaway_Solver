# Using PETSc with HYPRE support (testhypre)

## Problem

The default PETSc build pointed to by this repo's CMake config
(`~/Pets_Installation/petsc`) was built **without HYPRE support**
(`petscconf.h` has no `PETSC_HAVE_HYPRE`), so `testhypre` failed with:

```
This PETSc build has no hypre support (PETSC_HAVE_HYPRE undefined) -- cannot run the boomeramg preconditioner.
```

Several other PETSc installs on this machine do have HYPRE, but most of them
(conda envs like `fenics-env`) are built against **MPICH**, while this
project's build and the system `mpirun` use **Open MPI**. Mixing the two
causes segfaults — do not use the conda/miniconda PETSc installs for this.

## Fix: use the native system install `/usr/local/ff-petsc/r`

PETSc 3.22.2, `PETSC_HAVE_HYPRE 1`, built against **system Open MPI** — matches
the rest of the toolchain, no conda involved.

```bash
 unset LD_LIBRARY_PATH



cd /home/francesco-virgulti/Desktop/DREAM/build

# Reconfigure CMake to point at the HYPRE-enabled PETSc
cmake -DPETSC_DIR=/usr/local/ff-petsc/r -DPETSC_ARCH= .

# Rebuild the target
make testhypre

# Run with the system mpirun; unset LD_LIBRARY_PATH to avoid any
# conda libs (e.g. from fenics-env) leaking in and clashing
cd iface
env -u LD_LIBRARY_PATH mpirun -n 2 ./testhypre
```

Expected output includes both solves converging:

```
hypre/boomeramg GMRES: its = 61, converged (reason 2), time 0.565162 s
bjacobi/ilu GMRES:     its = 21, converged (reason 2), time 0.0349619 s
||A*x_amg - b|| / ||b|| = 3.69e-11  (PASS)
```

## Notes / caveats

- This only changes the CMake cache in `Desktop/DREAM/build` (via
  `PETSC_DIR`/`PETSC_ARCH`), not anything in the `Runaway_Solver` source tree.
  Other build directories will still default to whatever `PETSC_DIR` they were
  configured with.
- If `~/Pets_Installation/petsc` should gain HYPRE support long-term instead,
  it needs to be reconfigured/rebuilt with `--download-hypre` (or pointed at
  system `libhypre-dev`), then reconfigure this project against it again.
- Do not mix `/usr/local/ff-petsc/r` (or any Open-MPI-based PETSc) with
  conda's MPICH-based `mpirun`/`mpiexec`, and vice versa — always match the
  MPI implementation PETSc was built against with the `mpirun` used to launch.
