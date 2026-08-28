"""
stealth_optimizer.py
=====================
Gradient-free geometry optimization: sweep wing leading-edge sweep angle,
V-tail cant angle, and nose length to MINIMIZE mean monostatic RCS
(azimuth-averaged, X-band 10 GHz, full PO+PTD+DoubleBounce physics).

"Mean RCS" is averaged in the LINEAR domain (radar-cross-section convention)
then converted back to dB for reporting/optimizing, since averaging dBsm
directly is not physically meaningful:
    sigma_mean = mean(10**(sigma_dbsm/10)) ;  report 10*log10(sigma_mean)

Optimizer: scipy.optimize.differential_evolution (gradient-free, global,
handles the non-convex/non-smooth RCS landscape -- gradients of a faceted
PO/PTD model are not well-defined anyway).

Run: python stealth_optimizer.py
Output: output_ppt/12_optimization/*.png, printed best parameter set.
"""
import os, sys, time
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import stealth_aircraft
from solvers import DoubleBounce, PTDSolver, POAmplitudeSolver
from plot_style import apply_style, style_axis, PALETTE

apply_style()
OUT = './output_ppt/12_optimization'
os.makedirs(OUT, exist_ok=True)
FREQ_HZ = 10e9

AZ_EVAL = np.linspace(0, 360, 24, endpoint=False)   # optimizer's own az sweep (cheap, subdiv=1)
BOUNDS = [(30.0, 70.0), (10.0, 60.0), (0.5, 4.0)]   # sweep_deg, cant_deg, nose_length
NAMES = ['sweep_deg', 'cant_deg', 'nose_length']


def mean_rcs_dbsm(vals_dbsm):
    lin = 10.0 ** (np.asarray(vals_dbsm) / 10.0)
    return 10.0 * np.log10(np.mean(lin))


def mean_rcs_for_params(sweep_deg, cant_deg, nose_length, subdiv=1, az=AZ_EVAL, SolverCls=DoubleBounce):
    v, f = stealth_aircraft(subdiv=subdiv, sweep_deg=sweep_deg, cant_deg=cant_deg, nose_length=nose_length)
    s = SolverCls(v, f, FREQ_HZ)
    return mean_rcs_dbsm([s.monostatic_rcs_dbsm(a) for a in az])


history = []


def objective(params):
    val = mean_rcs_for_params(*params)
    return val


def track(xk, convergence):
    history.append(objective(xk))


def run_optimization():
    t0 = time.time()
    # baseline = bit-exact hardcoded geometry (sweep_deg=None path)
    v0, f0 = stealth_aircraft(subdiv=1)
    baseline = mean_rcs_dbsm([DoubleBounce(v0, f0, FREQ_HZ).monostatic_rcs_dbsm(a) for a in AZ_EVAL])
    print(f"Baseline stealth geometry: mean RCS = {baseline:.3f} dBsm")

    result = differential_evolution(objective, BOUNDS, seed=42, maxiter=40, popsize=20,
                                     tol=1e-4, callback=track, polish=True, workers=1)
    dt = time.time() - t0
    best = dict(zip(NAMES, result.x))
    print(f"\nOptimizer done in {dt:.1f}s, {result.nfev} evaluations")
    print(f"Best params: sweep_deg={best['sweep_deg']:.2f}  cant_deg={best['cant_deg']:.2f}  "
          f"nose_length={best['nose_length']:.2f}")
    print(f"Best mean RCS = {result.fun:.3f} dBsm  (baseline was {baseline:.3f} dBsm, "
          f"delta={result.fun - baseline:+.3f} dB)")

    # -- Plot 1: DE convergence history --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, color=PALETTE['STLTH'], lw=2.0, marker='o', ms=3)
    ax.axhline(baseline, color=PALETTE['BASELINE'], ls=':', lw=1.4, label=f'Baseline ({baseline:.2f} dBsm)')
    style_axis(ax, title='Differential evolution convergence — mean RCS vs generation',
               xlabel='Generation', ylabel='Best mean RCS so far (dBsm)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/optimizer_convergence.png', dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}/optimizer_convergence.png")

    # -- Plot 2: baseline vs optimized azimuth sweep --
    v_opt, f_opt = stealth_aircraft(subdiv=1, sweep_deg=best['sweep_deg'],
                                     cant_deg=best['cant_deg'], nose_length=best['nose_length'])
    s_opt = DoubleBounce(v_opt, f_opt, FREQ_HZ)
    s_base = DoubleBounce(v0, f0, FREQ_HZ)
    az_fine = np.linspace(0, 360, 181, endpoint=False)
    rcs_base = [s_base.monostatic_rcs_dbsm(a) for a in az_fine]
    rcs_opt = [s_opt.monostatic_rcs_dbsm(a) for a in az_fine]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(az_fine, rcs_base, color=PALETTE['BASELINE'], lw=1.8, label=f'Baseline (mean={baseline:.2f} dBsm)')
    ax.plot(az_fine, rcs_opt, color=PALETTE['STLTH'], lw=2.0, label=f"Optimized (mean={result.fun:.2f} dBsm)")
    style_axis(ax, title='Baseline vs DE-optimized stealth geometry — azimuth RCS sweep, X-band',
               xlabel='Azimuth (deg)', ylabel='RCS (dBsm)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/optimizer_baseline_vs_optimized.png', dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}/optimizer_baseline_vs_optimized.png")

    return best, result.fun, baseline


def run_convergence(best):
    """Mesh convergence of the mean-RCS metric at the optimum parameter set."""
    TIERS = ([(s, DoubleBounce) for s in [1, 2, 4, 8]] +
             [(s, PTDSolver) for s in [16, 32, 64]] +
             [(s, POAmplitudeSolver) for s in [150, 300, 600]])
    facets, means = [], []
    for subdiv, SolverCls in TIERS:
        t0 = time.time()
        val = mean_rcs_for_params(best['sweep_deg'], best['cant_deg'], best['nose_length'],
                                   subdiv=subdiv, az=np.linspace(0, 360, 24, endpoint=False),
                                   SolverCls=SolverCls)
        v, f = stealth_aircraft(subdiv=subdiv, sweep_deg=best['sweep_deg'],
                                 cant_deg=best['cant_deg'], nose_length=best['nose_length'])
        facets.append(len(f))
        means.append(val)
        print(f"subdiv={subdiv:<5} facets={len(f):<10,} {SolverCls.__name__:<18} "
              f"mean_rcs={val:7.3f} dBsm ({time.time()-t0:.1f}s)")

    assert all(np.isfinite(means))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(facets, means, marker='o', ms=6, lw=2.2, color=PALETTE['ACCENT'])
    ax.set_xscale('log')
    ax.axhline(means[-1], color=PALETTE['DIM'], ls=':', lw=1, label=f'Converged ({means[-1]:.2f} dBsm)')
    style_axis(ax, title='Mesh convergence at optimum parameter set',
               xlabel='Facet count (log scale)', ylabel='Mean RCS (dBsm)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/optimizer_mesh_convergence.png', dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}/optimizer_mesh_convergence.png")


if __name__ == '__main__':
    best, best_val, baseline = run_optimization()
    run_convergence(best)
    print(f"\nFINAL: sweep_deg={best['sweep_deg']:.2f} cant_deg={best['cant_deg']:.2f} "
          f"nose_length={best['nose_length']:.2f} -> mean RCS {best_val:.3f} dBsm "
          f"({best_val - baseline:+.3f} dB vs baseline)")
