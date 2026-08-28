"""
advanced_convergence_study.py
==============================
Mesh convergence for the bistatic and polarimetric RCS layers: does the
cross-pol (HV/VH) and bistatic RCS also stabilise as facets shrink, the
same question convergence_study.py asks of monostatic RCS?

Same three-tier solver strategy as convergence_study.py, but the
polarimetric double-bounce path (cascaded Jones matrices per facet pair)
carries extra per-pair cost on top of the already-O(n^2) DoubleBounce loop,
so its tier is capped lower (empirically timed below).

Run: python advanced_convergence_study.py
Output: output_ppt/10_convergence/mesh_convergence_bistatic.png
        output_ppt/10_convergence/mesh_convergence_polarimetric.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce, PTDSolver, POAmplitudeSolver
from plot_style import apply_style, style_axis, PALETTE

apply_style()
OUT = './output_ppt/10_convergence'
os.makedirs(OUT, exist_ok=True)

FREQ_HZ = 10e9
GEOMETRIES = {'Conventional': conventional_aircraft, 'Stealth': stealth_aircraft}

# Polarimetric double-bounce (cascaded Jones matrices, O(n^2) with extra
# per-pair cost) is far more expensive than scalar DoubleBounce -- capped
# lower. PO-only polarimetric is fully vectorised, scales like the basic study.
BISTATIC_TIERS = (
    [(s, DoubleBounce)      for s in [1, 2, 4, 8, 16]] +
    [(s, PTDSolver)         for s in [32, 64, 128]] +
    [(s, POAmplitudeSolver) for s in [300, 600, 1200]]
)
POLARIMETRIC_TIERS = (
    [(s, DoubleBounce)      for s in [1, 2, 4, 8]] +
    [(s, PTDSolver)         for s in [16, 32, 64]] +
    [(s, POAmplitudeSolver) for s in [128, 300, 600]]
)

# Fixed geometry for both sweeps: tx nose-on, rx at a generic bistatic
# offset; polarimetric probed at 45deg (conventional's corner-reflector
# activation angle) so the cross-pol convergence question is meaningful.
TX, RX = (0.0, 0.0), (45.0, 15.0)
POL_ANGLE = 45.0


def run_bistatic():
    results = {geo: {'facets': [], 'rcs': []} for geo in GEOMETRIES}
    for geo_name, builder in GEOMETRIES.items():
        for subdiv, SolverCls in BISTATIC_TIERS:
            v, f = builder(subdiv=subdiv)
            t0 = time.time()
            solver = SolverCls(v, f, FREQ_HZ)
            rcs = solver.bistatic_rcs_dbsm(TX[0], TX[1], RX[0], RX[1])
            dt = time.time() - t0
            results[geo_name]['facets'].append(len(f))
            results[geo_name]['rcs'].append(rcs)
            print(f"{geo_name:<14} subdiv={subdiv:<6} facets={len(f):<10,} "
                  f"{SolverCls.__name__:<18} bistatic_rcs={rcs:7.3f} dBsm ({dt:.2f}s)")

    for geo_name in GEOMETRIES:
        assert results[geo_name]['facets'] == sorted(results[geo_name]['facets'])
        assert all(np.isfinite(results[geo_name]['rcs']))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for geo_name in GEOMETRIES:
        r = results[geo_name]
        color = PALETTE['CONV'] if geo_name == 'Conventional' else PALETTE['STLTH']
        ax.plot(r['facets'], r['rcs'], marker='o', ms=5, lw=2.2, color=color, label=geo_name)
        ax.axhline(r['rcs'][-1], color=color, ls=':', lw=1, alpha=0.5)
    ax.set_xscale('log')
    style_axis(ax, title=f'Bistatic RCS convergence (tx=nose-on, rx=45deg az/15deg el)',
               xlabel='Facet count (log scale)', ylabel='Bistatic RCS (dBsm)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/mesh_convergence_bistatic.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}/mesh_convergence_bistatic.png\n")


def run_polarimetric():
    results = {geo: {'facets': [], 'HH': [], 'HV': []} for geo in GEOMETRIES}
    for geo_name, builder in GEOMETRIES.items():
        for subdiv, SolverCls in POLARIMETRIC_TIERS:
            v, f = builder(subdiv=subdiv)
            t0 = time.time()
            solver = SolverCls(v, f, FREQ_HZ)
            pol = solver.polarimetric_rcs_dbsm(POL_ANGLE, 0.0)
            dt = time.time() - t0
            results[geo_name]['facets'].append(len(f))
            results[geo_name]['HH'].append(pol['HH'])
            results[geo_name]['HV'].append(pol['HV'])
            print(f"{geo_name:<14} subdiv={subdiv:<6} facets={len(f):<10,} "
                  f"{SolverCls.__name__:<18} HH={pol['HH']:7.2f}  HV={pol['HV']:7.2f} dBsm ({dt:.2f}s)")

    for geo_name in GEOMETRIES:
        assert results[geo_name]['facets'] == sorted(results[geo_name]['facets'])
        assert all(np.isfinite(results[geo_name]['HH']))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, geo_name in zip(axes, GEOMETRIES):
        r = results[geo_name]
        ax.plot(r['facets'], r['HH'], marker='o', ms=5, lw=2.2, color=PALETTE['HH'], label='HH (co-pol)')
        ax.plot(r['facets'], r['HV'], marker='^', ms=5, lw=2.2, color=PALETTE['HV'], label='HV (cross-pol)')
        ax.set_xscale('log')
        style_axis(ax, title=f'{geo_name} — polarimetric convergence at {POL_ANGLE:.0f}deg',
                   xlabel='Facet count (log scale)', ylabel='RCS (dBsm)')
        ax.legend(fontsize=9)
    fig.suptitle('Polarimetric mesh convergence — does cross-pol (HV) also stabilise?',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/mesh_convergence_polarimetric.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}/mesh_convergence_polarimetric.png")


if __name__ == '__main__':
    run_bistatic()
    run_polarimetric()
