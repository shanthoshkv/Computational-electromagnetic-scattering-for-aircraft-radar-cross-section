"""
triple_bounce_convergence.py
=============================
Mesh convergence for triple-bounce (trihedral corner reflector) RCS.

TripleBounce is an O(n^3) Python loop -- it hard-caps far earlier than
double-bounce (O(n^2)) or PO (vectorised). This script pushes each tier as
far as actually finishes in reasonable time (empirically timed below) and
is honest about how few points that buys: 3-4 facet counts for the scalar
sweep, 2-3 for polarimetric. A short convergence run is still a real answer
to "does it converge" -- it just can't reach the millions-of-facets flat
line the PO study could.

Evaluated at each geometry's own peak trihedral angle (found by a coarse
sphere sweep in the original triple-bounce implementation):
  Conventional: az=85, el=60
  Stealth:      az=340, el=60

Run: python triple_bounce_convergence.py
Output: output_ppt/10_convergence/mesh_convergence_triple_bounce.png
        output_ppt/10_convergence/mesh_convergence_triple_bounce_polarimetric.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce, TripleBounce
from plot_style import apply_style, style_axis, PALETTE

apply_style()
OUT = './output_ppt/10_convergence'
os.makedirs(OUT, exist_ok=True)
FREQ_HZ = 10e9

# (builder, peak angle, color, max scalar subdiv, max polarimetric subdiv)
# -- caps chosen empirically so no single point exceeds ~50s.
GEOMETRIES = {
    'Conventional': dict(builder=conventional_aircraft, angle=(85.0, 60.0),
                          color=PALETTE['CONV'], scalar_subdivs=[1, 2, 3], pol_subdivs=[1, 2]),
    'Stealth':       dict(builder=stealth_aircraft, angle=(340.0, 60.0),
                          color=PALETTE['STLTH'], scalar_subdivs=[1, 2, 3, 4], pol_subdivs=[1, 2, 3]),
}
DB_REFERENCE_SUBDIVS = [1, 2, 4, 8]   # cheap double-bounce reference curve, same angle


def run_scalar():
    results = {name: {'db_facets': [], 'db_rcs': [], 'tb_facets': [], 'tb_rcs': []}
               for name in GEOMETRIES}

    for name, cfg in GEOMETRIES.items():
        az, el = cfg['angle']
        for s in DB_REFERENCE_SUBDIVS:
            v, f = cfg['builder'](subdiv=s)
            t0 = time.time()
            rcs = DoubleBounce(v, f, FREQ_HZ).monostatic_rcs_dbsm(az, el)
            results[name]['db_facets'].append(len(f))
            results[name]['db_rcs'].append(rcs)
            print(f"{name:<14} subdiv={s:<4} facets={len(f):<8,} DoubleBounce  "
                  f"rcs={rcs:7.2f} dBsm ({time.time()-t0:.1f}s)")
        for s in cfg['scalar_subdivs']:
            v, f = cfg['builder'](subdiv=s)
            t0 = time.time()
            rcs = TripleBounce(v, f, FREQ_HZ).monostatic_rcs_dbsm(az, el)
            results[name]['tb_facets'].append(len(f))
            results[name]['tb_rcs'].append(rcs)
            print(f"{name:<14} subdiv={s:<4} facets={len(f):<8,} TripleBounce  "
                  f"rcs={rcs:7.2f} dBsm ({time.time()-t0:.1f}s)")

    for name in GEOMETRIES:
        assert all(np.isfinite(results[name]['tb_rcs']))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, cfg) in zip(axes, GEOMETRIES.items()):
        r = results[name]
        ax.plot(r['db_facets'], r['db_rcs'], marker='s', ms=5, lw=1.8,
                color=cfg['color'], alpha=0.45, ls='--', label='Double bounce (reference)')
        ax.plot(r['tb_facets'], r['tb_rcs'], marker='o', ms=7, lw=2.4,
                color=cfg['color'], label='Triple bounce')
        ax.set_xscale('log')
        az, el = cfg['angle']
        style_axis(ax, title=f'{name} — triple-bounce at peak trihedral angle (az={az:.0f}, el={el:.0f})',
                   xlabel='Facet count (log scale)', ylabel='RCS (dBsm)')
        ax.legend(fontsize=8)
    fig.suptitle('Triple-bounce mesh convergence — O(n^3) caps this far earlier than PO/DB',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/mesh_convergence_triple_bounce.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}/mesh_convergence_triple_bounce.png\n")


def run_polarimetric():
    results = {name: {'facets': [], 'HH': [], 'HV': []} for name in GEOMETRIES}
    for name, cfg in GEOMETRIES.items():
        az, el = cfg['angle']
        for s in cfg['pol_subdivs']:
            v, f = cfg['builder'](subdiv=s)
            t0 = time.time()
            pol = TripleBounce(v, f, FREQ_HZ).polarimetric_rcs_dbsm(az, el)
            results[name]['facets'].append(len(f))
            results[name]['HH'].append(pol['HH'])
            results[name]['HV'].append(pol['HV'])
            print(f"{name:<14} subdiv={s:<4} facets={len(f):<8,} TripleBounce-pol  "
                  f"HH={pol['HH']:7.2f}  HV={pol['HV']:7.2f} dBsm ({time.time()-t0:.1f}s)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, cfg) in zip(axes, GEOMETRIES.items()):
        r = results[name]
        ax.plot(r['facets'], r['HH'], marker='o', ms=7, lw=2.2, color=PALETTE['HH'], label='HH (co-pol)')
        ax.plot(r['facets'], r['HV'], marker='^', ms=7, lw=2.2, color=PALETTE['HV'], label='HV (cross-pol)')
        ax.set_xscale('log')
        az, el = cfg['angle']
        style_axis(ax, title=f'{name} — triple-bounce polarimetric (az={az:.0f}, el={el:.0f})',
                   xlabel='Facet count (log scale)', ylabel='RCS (dBsm)')
        ax.legend(fontsize=9)
    fig.suptitle('Triple-bounce polarimetric convergence (few points -- O(n^3) x 2x2 matrix cost)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/mesh_convergence_triple_bounce_polarimetric.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}/mesh_convergence_triple_bounce_polarimetric.png")


if __name__ == '__main__':
    run_scalar()
    run_polarimetric()
