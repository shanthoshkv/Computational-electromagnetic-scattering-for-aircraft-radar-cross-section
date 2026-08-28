"""
sibc_comparison.py
===================
Surface Impedance Boundary Condition (Leontovich SIBC) vs bare PEC.
Compares finite-conductivity real aerospace materials against the PEC
assumption used everywhere else in this project.

Zs = (1+j) sqrt(pi*f*mu/sigma)   [SIBCMaterial.surface_impedance]
Gamma = (Zs - Z0) / (Zs + Z0)    [reflection_coefficient]

Materials (approx. room-temp conductivity, S/m):
  aluminum_2024   1.74e7   -- typical airframe skin
  stainless_304   1.45e6   -- fasteners/ducting
  titanium_6al4v  5.80e5   -- structural/hot-section
  cfrp_composite  3.0e4    -- modern composite skin, worst conductor here

Output: ./output_ppt/11_sibc/*.png
Run: python sibc_comparison.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce, SIBCMaterial
from plot_style import apply_style, style_axis, PALETTE

apply_style()
FREQ_HZ = 10e9
OUT = './output_ppt/11_sibc'
os.makedirs(OUT, exist_ok=True)

DIM = PALETTE['DIM']
GEOMETRIES = {'Conventional': (conventional_aircraft, PALETTE['CONV']),
              'Stealth': (stealth_aircraft, PALETTE['STLTH'])}

MATERIALS = {
    'PEC':            None,
    'Aluminum 2024':  SIBCMaterial.from_preset('aluminum_2024'),
    'Stainless 304':  SIBCMaterial.from_preset('stainless_304'),
    'Titanium 6Al4V': SIBCMaterial.from_preset('titanium_6al4v'),
    'CFRP composite': SIBCMaterial.from_preset('cfrp_composite'),
}
MAT_COLORS = ['#8892A6', '#00E5FF', '#39FF88', '#FFD040', '#FF5B35']


def azimuth_sweep_plot():
    """RCS vs azimuth, PEC vs each material, both aircraft."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    az = np.linspace(0, 360, 181, endpoint=False)
    for ax, (name, (builder, _)) in zip(axes, GEOMETRIES.items()):
        v, f = builder()
        for (mname, mat), color in zip(MATERIALS.items(), MAT_COLORS):
            s = DoubleBounce(v, f, FREQ_HZ, ram_model=mat)
            rcs = [s.monostatic_rcs_dbsm(a, 0.0) for a in az]
            lw = 2.4 if mat is None else 1.6
            ls = '-' if mat is None else '--'
            ax.plot(az, rcs, color=color, lw=lw, ls=ls, label=mname)
        style_axis(ax, title=f'{name} — SIBC vs PEC, X-band 10 GHz',
                   xlabel='Azimuth (deg)', ylabel='RCS (dBsm)')
        ax.legend(fontsize=7.5)
    fig.suptitle('Finite-conductivity skin (Leontovich SIBC) vs bare PEC — azimuth sweep',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/sibc_azimuth_sweep.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/sibc_azimuth_sweep.png')


def frequency_sweep_plot():
    """Reflection loss vs frequency, L-band through Ka-band, per material."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    freqs = np.linspace(1e9, 40e9, 200)
    for (mname, mat), color in zip(MATERIALS.items(), MAT_COLORS):
        if mat is None:
            continue
        loss = mat.reflection_loss_db(freqs)
        ax.plot(freqs / 1e9, loss, color=color, lw=2.0, label=mname)
    ax.axhline(0, color=DIM, ls=':', lw=1)
    for band, fc in [('L', 1.5), ('S', 3), ('C', 6), ('X', 10), ('Ku', 15), ('Ka', 33)]:
        ax.axvline(fc, color=DIM, lw=0.5, alpha=0.4)
        ax.text(fc, ax.get_ylim()[0] if ax.get_ylim()[0] else -0.1, band, fontsize=7, color=DIM, ha='center')
    style_axis(ax, title='Reflection loss vs frequency — real materials deviate more from PEC as f increases',
               xlabel='Frequency (GHz)', ylabel='Reflection loss (dB, 0 = perfect PEC)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/sibc_frequency_sweep.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/sibc_frequency_sweep.png')


def rcs_reduction_bars():
    """Bar chart: mean RCS drop from PEC baseline, per material per aircraft,
    averaged over a full azimuth sweep at X-band."""
    az = np.linspace(0, 360, 73, endpoint=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    x = np.arange(len(MATERIALS) - 1)
    for i, (name, (builder, color)) in enumerate(GEOMETRIES.items()):
        v, f = builder()
        pec_rcs = np.array([DoubleBounce(v, f, FREQ_HZ).monostatic_rcs_dbsm(a, 0.0) for a in az])
        deltas = []
        for mname, mat in MATERIALS.items():
            if mat is None:
                continue
            s = DoubleBounce(v, f, FREQ_HZ, ram_model=mat)
            mat_rcs = np.array([s.monostatic_rcs_dbsm(a, 0.0) for a in az])
            deltas.append(np.mean(mat_rcs - pec_rcs))
        ax.bar(x + (i - 0.5) * width, deltas, width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m in MATERIALS if m != 'PEC'], fontsize=8, rotation=15)
    style_axis(ax, title='Mean RCS drop vs bare PEC — azimuth-averaged, X-band 10 GHz',
               ylabel='Mean ΔRCS vs PEC (dB, negative = quieter)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/sibc_rcs_reduction_bars.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/sibc_rcs_reduction_bars.png')


if __name__ == '__main__':
    azimuth_sweep_plot()
    frequency_sweep_plot()
    rcs_reduction_bars()
    print('\nAll SIBC comparison plots saved to', OUT)
