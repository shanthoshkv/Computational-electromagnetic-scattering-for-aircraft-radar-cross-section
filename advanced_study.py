"""
advanced_study.py
==================
Bistatic and polarimetric RCS demonstrations, full physics
(PO+PTD+DoubleBounce) for conventional and stealth aircraft.

Output: ./output_ppt/09_bistatic_polarimetric/*.png

Run: python advanced_study.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce
from plot_style import apply_style, style_axis, PALETTE

apply_style()
FREQ_HZ = 10e9
OUT = './output_ppt/09_bistatic_polarimetric'
os.makedirs(OUT, exist_ok=True)

DIM = PALETTE['DIM']
C_CONV, C_STLTH = PALETTE['CONV'], PALETTE['STLTH']
C_HH, C_VV, C_HV, C_VH = PALETTE['HH'], PALETTE['VV'], PALETTE['HV'], PALETTE['VH']

GEOMETRIES = {'Conventional': (conventional_aircraft, C_CONV), 'Stealth': (stealth_aircraft, C_STLTH)}


def bistatic_sweep_plot():
    """Bistatic RCS vs bistatic angle, transmitter fixed nose-on (az=0,el=0),
    receiver azimuth swept 0-360. 0 deg = monostatic backscatter,
    180 deg = forward scatter (Babinet regime)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rx_az = np.linspace(0, 360, 181)
    for name, (builder, color) in GEOMETRIES.items():
        v, f = builder()
        s = DoubleBounce(v, f, FREQ_HZ)
        rcs = [s.bistatic_rcs_dbsm(0.0, 0.0, az, 0.0) for az in rx_az]
        ax.plot(rx_az, rcs, color=color, lw=2.2, label=name)
    ax.axvline(0, color=DIM, ls=':', lw=1)
    ax.axvline(180, color=PALETTE['WARN'], ls='--', lw=1.2)
    ax.text(3, ax.get_ylim()[1]*0.93, 'monostatic\n(backscatter)', fontsize=8, color=DIM)
    ax.text(183, ax.get_ylim()[1]*0.93, 'forward scatter\n(Babinet regime)', fontsize=8, color=PALETTE['WARN'])
    style_axis(ax, title='Bistatic RCS sweep — full physics (PO+PTD+DoubleBounce), X-band 10 GHz',
               xlabel='Receiver azimuth (deg) — transmitter fixed at nose-on (0 deg)',
               ylabel='Bistatic RCS (dBsm)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/bistatic_sweep_nose_tx.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/bistatic_sweep_nose_tx.png')


def bistatic_angle_sweep_plot():
    """RCS vs bistatic ANGLE beta (angle between tx and rx as seen from
    target), transmitter fixed broadside -- shows the classic bistatic
    RCS pattern shape independent of absolute geometry."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    beta = np.linspace(0, 180, 91)
    for name, (builder, color) in GEOMETRIES.items():
        v, f = builder()
        s = DoubleBounce(v, f, FREQ_HZ)
        rcs = [s.bistatic_rcs_dbsm(90.0, 0.0, 90.0 + b, 0.0) for b in beta]
        ax.plot(beta, rcs, color=color, lw=2.2, label=name)
    style_axis(ax, title='RCS vs bistatic angle — full physics, X-band 10 GHz',
               xlabel='Bistatic angle β (deg) — transmitter fixed at broadside (90 deg)',
               ylabel='Bistatic RCS (dBsm)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/bistatic_angle_sweep.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/bistatic_angle_sweep.png')


def polarimetric_azimuth_plot():
    """HH/VV/HV/VH vs azimuth, full physics, monostatic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    az = np.linspace(0, 360, 181, endpoint=False)
    for ax, (name, (builder, _)) in zip(axes, GEOMETRIES.items()):
        v, f = builder()
        s = DoubleBounce(v, f, FREQ_HZ)
        chans = {'HH': [], 'VV': [], 'HV': [], 'VH': []}
        for a in az:
            pol = s.polarimetric_rcs_dbsm(a, 0.0)
            for k in chans:
                chans[k].append(pol[k])
        ax.plot(az, chans['HH'], color=C_HH, lw=1.8, label='HH')
        ax.plot(az, chans['VV'], color=C_VV, lw=1.8, ls='--', label='VV')
        ax.plot(az, chans['HV'], color=C_HV, lw=1.6, label='HV (cross-pol)')
        ax.plot(az, chans['VH'], color=C_VH, lw=1.2, ls=':', label='VH (cross-pol)')
        style_axis(ax, title=f'{name} — polarimetric RCS, full physics',
                   xlabel='Azimuth (deg)', ylabel='RCS (dBsm)')
        ax.legend(fontsize=8)
        ax.set_ylim(-30, None)
    fig.suptitle('Polarimetric azimuth sweep — HH/VV co-pol vs HV/VH cross-pol',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/polarimetric_azimuth.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/polarimetric_azimuth.png')


def polarimetric_signature_bars():
    """Bar chart: HH/VV/HV/VH at nose-on, quarter-on, broadside for both
    aircraft -- shows the corner-reflector depolarisation signature."""
    angles = {'Nose-on (0deg)': 0.0, 'Quarter-on (45deg)': 45.0, 'Broadside (90deg)': 90.0}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.2
    for ax, (name, (builder, _)) in zip(axes, GEOMETRIES.items()):
        v, f = builder()
        s = DoubleBounce(v, f, FREQ_HZ)
        x = np.arange(len(angles))
        for i, chan in enumerate(['HH', 'VV', 'HV', 'VH']):
            color = {'HH': C_HH, 'VV': C_VV, 'HV': C_HV, 'VH': C_VH}[chan]
            vals = [s.polarimetric_rcs_dbsm(a, 0.0)[chan] for a in angles.values()]
            vals = [max(v, -30) for v in vals]   # floor for display
            ax.bar(x + (i-1.5)*width, vals, width, label=chan, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(angles.keys(), fontsize=8)
        style_axis(ax, title=f'{name} — polarimetric signature by aspect', ylabel='RCS (dBsm)')
        ax.legend(fontsize=8)
    fig.suptitle('Polarimetric signature — corner-reflector depolarisation by aspect',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/polarimetric_signature_bars.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/polarimetric_signature_bars.png')


if __name__ == '__main__':
    bistatic_sweep_plot()
    bistatic_angle_sweep_plot()
    polarimetric_azimuth_plot()
    polarimetric_signature_bars()
    print('\nAll advanced-study plots saved to', OUT)
