#!/usr/bin/env python3
"""
Physics checks of a kinetic DREAM output (f_hot and f_re), independent of
how many ranks computed it.

For every output file given:

  1. particle balance   N = sum_r dr int f V' dp dxi0 (the density moment of
                        DREAM, times the radial volume element), of f_hot,
                        f_re and both, at the first and the last time.
                        Without transport (DBB=0 AR_FACTOR=0) and without
                        sources the change must be at round-off level: the
                        f_hot -> f_re boundary at p = p_re is conservative.
                        With transport the change is the loss through the
                        edge (BC_F_0): it must be <= 0, and the per-radius
                        changes show where particles moved.
  2. positivity         fraction of negative cells and most negative value
                        relative to max f, at the last time.
  3. trapping           (toroidal only) on the negative trapped cells
                        (xi0 < 0 with V' = 0) f(xi0) must equal f at -xi0,
                        linearly interpolated between the two closest cell
                        centres (FVM::BC::PXiInternalTrapping): max relative
                        violation per grid.
  4. p = p_re interface f_hot in its last p cell against f_re in its first
                        p cell, f_re interpolated to the f_hot pitch
                        centres: ratio of the pitch integrals per radius
                        (close to 1 when f is continuous; the two cell
                        centres are half a cell apart on either side).

Options:

  --equilibrium   5. (E = 0 run, no transport) relative change of f_hot
                  between the first and the last time, and of its density
                  and of the energy moment: a Maxwellian at T_cold must stay.
  --compare REF   6. relative difference of f_hot, f_re and their density
                  profiles against the output REF on the same p-xi grids
                  (cylindrical limit: thin toroidal against cylinder).

Usage:

    python3 check_physics.py output_toroidal_CQ_t20ms_nr44_L1.h5
    python3 check_physics.py out_E0_notransport.h5 --equilibrium
    python3 check_physics.py out_thin_notransport.h5 --compare out_cylinder_notransport.h5
"""

import argparse
import sys

import h5py
import numpy as np

GRIDS = (("f_hot", "hottail"), ("f_re", "runaway"))


def load(path):
    """Grids and the first/last f of an output file."""
    out = {"path": path}
    with h5py.File(path, "r") as h:
        g = h["grid"]
        out["t"] = g["t"][:]
        out["r"] = g["r"][:]
        out["dr"] = g["dr"][:]
        out["VpVol"] = g["VpVol"][:]
        geo = g.get("geometry")
        out["xiT"] = geo["xi0TrappedBoundary"][:] if geo is not None and "xi0TrappedBoundary" in geo else None
        for q, name in GRIDS:
            if q not in h["eqsys"] or name not in g:
                continue
            mg = g[name]
            f = h["eqsys"][q]
            out[q] = dict(
                f0=np.asarray(f[0], float), f1=np.asarray(f[-1], float),
                Vp=mg["Vprime"][:], p=mg["p1"][:], xi=mg["p2"][:],
                dp=mg["dp1"][:], dxi=mg["dp2"][:])
    return out


def density_per_radius(d, f):
    """sum over p, xi of f V' dp dxi0 for each radius (= n(r) V'_vol)."""
    w = d["Vp"] * d["dxi"][None, :, None] * d["dp"][None, None, :]
    return np.sum(f * w, axis=(1, 2))


def check_balance(o):
    print("  1. particle balance  N = sum_r dr int f V' dp dxi0  (first -> last time)")
    tot0 = tot1 = 0.0
    per_r = 0.0
    for q, _ in GRIDS:
        if q not in o:
            continue
        d = o[q]
        n0 = density_per_radius(d, d["f0"]) * o["dr"]
        n1 = density_per_radius(d, d["f1"]) * o["dr"]
        tot0 += n0.sum()
        tot1 += n1.sum()
        per_r = per_r + (n1 - n0)
        print(f"     {q:6}  N0 = {n0.sum():.6e}  N1 = {n1.sum():.6e}  change = {n1.sum() - n0.sum():+.3e}")
    rel = (tot1 - tot0) / tot0
    print(f"     total   N0 = {tot0:.6e}  N1 = {tot1:.6e}  relative change = {rel:+.3e}")
    k = np.argmax(np.abs(per_r))
    print(f"     largest change at radius {k} (r = {o['r'][k]:.3f} m): {per_r[k] / tot0:+.3e} of N0;"
          f" edge radius: {per_r[-1] / tot0:+.3e}")
    return rel


def check_positivity(o):
    print("  2. positivity (last time)")
    for q, _ in GRIDS:
        if q not in o:
            continue
        f = o[q]["f1"]
        neg = f < 0
        fmax = np.abs(f).max()
        worst = f.min() / fmax if fmax > 0 else 0.0
        print(f"     {q:6}  negative cells {100 * neg.mean():6.2f} %   min f / max |f| = {worst:+.2e}")


def check_trapping(o):
    if o["xiT"] is None or np.all(o["xiT"] == 0):
        print("  3. trapping: no trapped region (cylindrical), skipped")
        return
    print("  3. trapping: f(xi0) = f(-xi0) on the negative trapped cells (last time)")
    for q, _ in GRIDS:
        if q not in o:
            continue
        d = o[q]
        f, xi, Vp = d["f1"], d["xi"], d["Vp"]
        worst, ncells = 0.0, 0
        for ir in range(f.shape[0]):
            fmax = np.abs(f[ir]).max()
            if fmax == 0:
                continue
            for j in np.nonzero((xi < 0) & (Vp[ir].max(axis=1) == 0))[0]:
                mirrored = np.array([np.interp(-xi[j], xi, f[ir, :, i]) for i in range(f.shape[2])])
                worst = max(worst, np.abs(f[ir, j] - mirrored).max() / fmax)
                ncells += f.shape[2]
        print(f"     {q:6}  {ncells} trapped cells, max |f(xi0) - f(-xi0)| / max f = {worst:.2e}")


def check_interface(o):
    if "f_hot" not in o or "f_re" not in o:
        return
    print("  4. p = p_re interface: int f_re(p_min) dxi0 / int f_hot(p_max) dxi0 per radius (last time)")
    h, r = o["f_hot"], o["f_re"]
    ratios = []
    for ir in range(h["f1"].shape[0]):
        fh = h["f1"][ir, :, -1]
        fr = np.interp(h["xi"], r["xi"], r["f1"][ir, :, 0])
        sh = np.sum(fh * h["dxi"])
        ratios.append(np.sum(fr * h["dxi"]) / sh if sh != 0 else np.nan)
    ratios = np.array(ratios)
    print(f"     p: f_hot cell {h['p'][-1]:.4f}, f_re cell {r['p'][0]:.4f}")
    print("     ratio: " + " ".join(f"{x:.3f}" for x in ratios))


def check_equilibrium(o):
    print("  5. equilibrium (E = 0): change of f_hot and its moments, first -> last time")
    d = o["f_hot"]
    f0, f1 = d["f0"], d["f1"]
    df = np.abs(f1 - f0).max() / np.abs(f0).max()
    n0, n1 = density_per_radius(d, f0), density_per_radius(d, f1)
    gamma = np.sqrt(1 + d["p"] ** 2)[None, None, :]
    w0 = density_per_radius(d, f0 * (gamma - 1))
    w1 = density_per_radius(d, f1 * (gamma - 1))
    print(f"     max |f1 - f0| / max f0 = {df:.2e}")
    print(f"     density: max relative change over r = {np.max(np.abs(n1 / n0 - 1)):.2e}")
    print(f"     kinetic energy: max relative change over r = {np.max(np.abs(w1 / w0 - 1)):.2e}")


def check_compare(o, ref):
    print(f"  6. against {ref['path']}")
    for q, _ in GRIDS:
        if q not in o or q not in ref:
            continue
        a, b = ref[q]["f1"], o[q]["f1"]
        if a.shape != b.shape:
            print(f"     {q:6}  shapes differ {a.shape} vs {b.shape}")
            continue
        df = np.abs(a - b).max() / np.abs(a).max()
        na = density_per_radius(ref[q], a) / ref["VpVol"]
        nb = density_per_radius(o[q], b) / o["VpVol"]
        dn = np.max(np.abs(nb - na) / np.maximum(np.abs(na), 1e-300))
        print(f"     {q:6}  max |df| / max f = {df:.2e}   density profile: max relative difference = {dn:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("outputs", nargs="+", help="DREAM output files")
    ap.add_argument("--equilibrium", action="store_true", help="E = 0 run: f_hot must not change")
    ap.add_argument("--compare", metavar="REF", help="compare f against this output (same grids)")
    args = ap.parse_args()

    ref = load(args.compare) if args.compare else None
    for path in args.outputs:
        o = load(path)
        print(f"\n=== {path}   (t = {o['t'][0]:.3e} -> {o['t'][-1]:.3e} s, Nr = {o['r'].size})")
        check_balance(o)
        check_positivity(o)
        check_trapping(o)
        check_interface(o)
        if args.equilibrium:
            check_equilibrium(o)
        if ref is not None:
            check_compare(o, ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
