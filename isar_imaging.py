"""
isar_imaging.py
================
ISAR (Inverse Synthetic Aperture Radar) imaging: take a frequency x angle
sweep of the SAME coherent full-physics amplitude used in
wideband_impulse_response.py, and 2D inverse-FFT it into a range/cross-range
image -- the standard small-aperture FFT-based ISAR algorithm (valid because
the angular aperture below is kept small; a wide-aperture image needs polar
reformatting/backprojection, out of scope here).

  H[angle, freq] = f_PO + f_edge + f_DB    (coherent complex, per bin)
  image = IFFT2{ window2D . H }
  range axis      <- frequency axis, resolution dR  = c/(2B)
  cross-range axis <- angle axis,    resolution dCR = lambda_c/(2*aperture_rad)
    (small-angle rotational-Doppler approximation, standard for narrow ISAR apertures)

Run: python isar_imaging.py
Output: output_ppt/14_isar/*.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce, PTDSolver, POAmplitudeSolver, C_LIGHT
from wideband_impulse_response import coherent_amplitude
from plot_style import apply_style, PALETTE

apply_style()
OUT = './output_ppt/14_isar'
os.makedirs(OUT, exist_ok=True)

F_LOW, F_HIGH, N_FREQ = 8e9, 12e9, 128     # X-band, 4 GHz bandwidth
CENTER_AZ, APERTURE_DEG, N_ANG = 90.0, 10.0, 128   # broadside +/- 5 deg (small-angle ISAR aperture)
FREQ_CENTER = 0.5 * (F_LOW + F_HIGH)
LAMBDA_C = C_LIGHT / FREQ_CENTER

GEOMETRIES = {'Conventional': (conventional_aircraft, PALETTE['CONV']),
              'Stealth': (stealth_aircraft, PALETTE['STLTH'])}


def sweep_2d(builder, freqs, angles, subdiv=2, SolverCls=DoubleBounce):
    v, f = builder(subdiv=subdiv)
    H = np.empty((len(angles), len(freqs)), dtype=complex)
    for ai, az in enumerate(angles):
        for fi, fr in enumerate(freqs):
            s = SolverCls(v, f, fr)
            shat = s._shat(az, 0.0)
            H[ai, fi] = coherent_amplitude(s, shat)
    return H


def isar_image(H, freqs, angles):
    win = np.outer(np.hanning(len(angles)), np.hanning(len(freqs)))
    img = np.fft.fftshift(np.fft.ifft2(win * H))
    dR = C_LIGHT / (2.0 * (freqs[-1] - freqs[0]))
    aperture_rad = np.radians(angles[-1] - angles[0])
    dCR = LAMBDA_C / (2.0 * aperture_rad)
    R = (np.arange(len(freqs)) - len(freqs) // 2) * dR
    CR = (np.arange(len(angles)) - len(angles) // 2) * dCR
    return R, CR, np.abs(img)


def image_plot():
    freqs = np.linspace(F_LOW, F_HIGH, N_FREQ)
    angles = np.linspace(CENTER_AZ - APERTURE_DEG / 2, CENTER_AZ + APERTURE_DEG / 2, N_ANG)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (name, (builder, _)) in zip(axes, GEOMETRIES.items()):
        t0 = time.time()
        H = sweep_2d(builder, freqs, angles)
        R, CR, mag = isar_image(H, freqs, angles)
        mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-30) + 1e-4)
        im = ax.imshow(mag_db, extent=[R.min(), R.max(), CR.min(), CR.max()],
                        origin='lower', aspect='auto', cmap='inferno', vmin=-40, vmax=0)
        ax.set_title(f'{name} — ISAR image ({time.time()-t0:.1f}s)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Cross-range (m)')
        fig.colorbar(im, ax=ax, label='Normalized magnitude (dB)', shrink=0.85)
    fig.suptitle(f'ISAR imaging — X-band 8-12 GHz, broadside +/-{APERTURE_DEG/2:.0f} deg aperture',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/isar_images.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}/isar_images.png')


def convergence_study():
    """Mesh convergence of ISAR image total energy vs facet count -- small
    frequency/angle grid to keep the O(n^2) DoubleBounce tier bounded."""
    freqs = np.linspace(F_LOW, F_HIGH, 8)
    angles = np.linspace(CENTER_AZ - APERTURE_DEG / 2, CENTER_AZ + APERTURE_DEG / 2, 8)
    TIERS = ([(s, DoubleBounce) for s in [1, 2, 4, 8]] +
             [(s, PTDSolver) for s in [16, 32, 64]] +
             [(s, POAmplitudeSolver) for s in [150, 300, 600]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, (builder, color)) in zip(axes, GEOMETRIES.items()):
        facets, energies = [], []
        for subdiv, SolverCls in TIERS:
            t0 = time.time()
            H = sweep_2d(builder, freqs, angles, subdiv=subdiv, SolverCls=SolverCls)
            v, f = builder(subdiv=subdiv)
            energy_db = 10.0 * np.log10(np.sum(np.abs(H) ** 2) + 1e-30)
            facets.append(len(f))
            energies.append(energy_db)
            print(f"{name:<14} subdiv={subdiv:<5} facets={len(f):<10,} {SolverCls.__name__:<18} "
                  f"image_energy={energy_db:7.2f} dB ({time.time()-t0:.1f}s)")
        ax.plot(facets, energies, marker='o', ms=6, lw=2.2, color=color)
        ax.set_xscale('log')
        ax.set_title(f'{name} — ISAR energy convergence', fontsize=11, fontweight='bold')
        ax.set_xlabel('Facet count (log scale)')
        ax.set_ylabel('Total image energy (dB)')
        ax.grid(True, alpha=0.35)
    fig.suptitle('Mesh convergence — ISAR frequency x angle sweep',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/isar_mesh_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}/isar_mesh_convergence.png')


if __name__ == '__main__':
    image_plot()
    convergence_study()
    print('\nAll ISAR imaging plots saved to', OUT)
