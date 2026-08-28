"""
optimizer_sensitivity.py
=========================
Why the DE optimizer landed on sweep=59.79, cant=19.95, nose_length=0.50 --
1D slices through the mean-RCS objective around the optimum (each parameter
swept alone, the other two held at their optimal value), plus a 2D
sweep x cant landscape (nose_length held at optimum) so the search space
itself is visible, not just the final point.

Run: python optimizer_sensitivity.py
Output: output_ppt/12_optimization/optimizer_sensitivity.png
        output_ppt/12_optimization/optimizer_landscape_heatmap.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stealth_optimizer import mean_rcs_for_params, BOUNDS
from plot_style import apply_style, style_axis, PALETTE

apply_style()
OUT = './output_ppt/12_optimization'
os.makedirs(OUT, exist_ok=True)

BEST = dict(sweep_deg=59.79, cant_deg=19.95, nose_length=0.50)
BASELINE = 51.58   # hand-designed baseline mean RCS, dBsm (stealth_optimizer.py run)
BEST_RCS = 48.44


def sensitivity_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [
        ('sweep_deg', 'Wing sweep $\\Lambda$ (deg)', BOUNDS[0]),
        ('cant_deg', 'V-tail cant $\\gamma$ (deg)', BOUNDS[1]),
        ('nose_length', 'Nose length $L_n$ (m)', BOUNDS[2]),
    ]
    for ax, (param, label, (lo, hi)) in zip(axes, specs):
        xs = np.linspace(lo, hi, 120)
        ys = []
        for x in xs:
            kw = dict(BEST)
            kw[param] = x
            ys.append(mean_rcs_for_params(**kw))
        ax.plot(xs, ys, color=PALETTE['STLTH'], lw=2.2, marker='o', ms=4)
        ax.axvline(BEST[param], color=PALETTE['ACCENT'], ls='--', lw=1.4,
                   label=f'optimum = {BEST[param]:.2f}')
        ax.axhline(BASELINE, color=PALETTE['BASELINE'], ls=':', lw=1.2, label=f'baseline ({BASELINE:.2f} dBsm)')
        style_axis(ax, title=f'Mean RCS vs {label}', xlabel=label, ylabel='Mean RCS (dBsm)')
        ax.legend(fontsize=8)
    fig.suptitle('Why these parameter values: 1D slices through the objective at the DE optimum',
                 fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(f'{OUT}/optimizer_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}/optimizer_sensitivity.png')
    return specs


def landscape_heatmap():
    sweeps = np.linspace(*BOUNDS[0], 16)
    cants = np.linspace(*BOUNDS[1], 70)
    grid = np.zeros((len(cants), len(sweeps)))
    for i, c in enumerate(cants):
        for j, s in enumerate(sweeps):
            grid[i, j] = mean_rcs_for_params(sweep_deg=s, cant_deg=c, nose_length=BEST['nose_length'])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='inferno',
                    extent=[sweeps[0], sweeps[-1], cants[0], cants[-1]])
    fig.colorbar(im, ax=ax, label='Mean RCS (dBsm)')
    ax.plot(BEST['sweep_deg'], BEST['cant_deg'], marker='*', ms=20, color=PALETTE['ACCENT'],
            markeredgecolor='black', markeredgewidth=0.8, label='DE optimum')
    style_axis(ax, title=f'Mean-RCS landscape, sweep x cant (nose_length={BEST["nose_length"]:.2f} m fixed)',
               xlabel='Wing sweep $\\Lambda$ (deg)', ylabel='V-tail cant $\\gamma$ (deg)')
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/optimizer_landscape_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}/optimizer_landscape_heatmap.png')
    return sweeps, cants, grid


if __name__ == '__main__':
    sensitivity_plot()
    landscape_heatmap()
    print('\nAll optimizer sensitivity plots saved to', OUT)
