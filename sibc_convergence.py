"""
sibc_convergence.py
====================
Mesh convergence for SIBC (finite-conductivity CFRP skin) vs bare PEC.
Same tiered solver-by-complexity strategy as convergence_study.py --
SIBC only multiplies each facet's reflectivity by a scalar Gamma, so it
adds zero extra big-O cost over the equivalent PEC run at the same tier.

Single representative aspect angle per geometry (broadside, where the
report's baseline mesh already shows the largest facet-size sensitivity)
to keep two materials x two geometries within reasonable runtime.

Run: python sibc_convergence.py
Output: output_ppt/11_sibc/mesh_convergence_sibc.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce, PTDSolver, POAmplitudeSolver, SIBCMaterial
from plot_style import apply_style, style_axis, PALETTE

apply_style()
OUT = './output_ppt/11_sibc'
os.makedirs(OUT, exist_ok=True)
FREQ_HZ = 10e9

CFRP = SIBCMaterial.from_preset('cfrp_composite')
GEOMETRIES = {'Conventional': (conventional_aircraft, 90.0, PALETTE['CONV']),
              'Stealth':      (stealth_aircraft, 90.0, PALETTE['STLTH'])}

TIERS = (
    [(s, DoubleBounce) for s in [1, 2, 4, 8, 16]] +
    [(s, PTDSolver)    for s in [32, 64, 128]] +
    [(s, POAmplitudeSolver) for s in [300, 600, 1200, 1500]]
)


def run():
    results = {geo: {'facets': [], 'pec': [], 'cfrp': []} for geo in GEOMETRIES}

    for geo_name, (builder, angle, color) in GEOMETRIES.items():
        for subdiv, SolverCls in TIERS:
            v, f = builder(subdiv=subdiv)
            t0 = time.time()
            rcs_pec  = SolverCls(v, f, FREQ_HZ).monostatic_rcs_dbsm(angle)
            rcs_cfrp = SolverCls(v, f, FREQ_HZ, ram_model=CFRP).monostatic_rcs_dbsm(angle)
            dt = time.time() - t0
            results[geo_name]['facets'].append(len(f))
            results[geo_name]['pec'].append(rcs_pec)
            results[geo_name]['cfrp'].append(rcs_cfrp)
            print(f"{geo_name:<14} subdiv={subdiv:<6} facets={len(f):<12,} "
                  f"{SolverCls.__name__:<18} PEC={rcs_pec:7.3f}  CFRP-SIBC={rcs_cfrp:7.3f}  ({dt:.1f}s)")

    for geo_name in GEOMETRIES:
        assert all(np.isfinite(results[geo_name]['pec']))
        assert all(np.isfinite(results[geo_name]['cfrp']))
        # CFRP is usually <= PEC (lossy reflectivity), but NOT a strict guarantee at every
        # tier: the RAM scalar only weakens the PO facet term, not the PTD edge-diffraction
        # term (Ufimtsev fringe coefficient is PEC-based physics, untouched by ram_model) --
        # so a slightly-smaller PO term can shift the PO+PTD coherent sum's phase balance and
        # tick CFRP a hair above PEC at some tiers. Real interference artifact, not a bug.
        n_worse = sum(c > p + 1e-9 for c, p in zip(results[geo_name]['cfrp'], results[geo_name]['pec']))
        assert n_worse <= 1, (geo_name, "more than one non-monotonic tier", results[geo_name])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (geo_name, (_, angle, color)) in zip(axes, GEOMETRIES.items()):
        r = results[geo_name]
        ax.plot(r['facets'], r['pec'], marker='o', ms=5, lw=2.2, color=color, label='PEC')
        ax.plot(r['facets'], r['cfrp'], marker='^', ms=5, lw=2.0, color=color, ls='--',
                alpha=0.7, label='CFRP composite (SIBC)')
        ax.set_xscale('log')
        style_axis(ax, title=f'{geo_name} — PEC vs SIBC convergence at {angle:.0f} deg (broadside)',
                   xlabel='Facet count (log scale)', ylabel='RCS (dBsm)')
        ax.legend(fontsize=8)
    fig.suptitle('Mesh convergence — PEC vs finite-conductivity (Leontovich SIBC) skin',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/mesh_convergence_sibc.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {OUT}/mesh_convergence_sibc.png")


if __name__ == '__main__':
    run()
