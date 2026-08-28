"""
wideband_impulse_response.py
=============================
Wideband time-domain impulse response (range profile) instead of
single-frequency RCS. Sweeps the SAME full-physics coherent complex
scatter amplitude (PO + PTD edge + double-bounce) used everywhere else
in this project across a band of frequencies, then inverse-FFTs the
frequency response into a range profile -- the standard stepped-frequency
radar processing chain used to resolve individual scattering centers
(nose, wing leading edge, tail) along the target's line of sight.

H(f) = f_PO(f) + f_edge(f) + f_DB(f)          [coherent, complex]
h(t) = IFFT{ W(f) . H(f) }                    [W = Hann window, sidelobe control]
R    = c.t / 2                                [round-trip range axis]
Range resolution: dR = c / (2B)

Run: python wideband_impulse_response.py
Output: output_ppt/13_wideband/*.png
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from geometries import conventional_aircraft, stealth_aircraft
from solvers import DoubleBounce, PTDSolver, POAmplitudeSolver, C_LIGHT
from plot_style import apply_style, style_axis, PALETTE

apply_style()
OUT = './output_ppt/13_wideband'
os.makedirs(OUT, exist_ok=True)

F_LOW, F_HIGH, N_FREQ = 2e9, 18e9, 512   # S-band through Ku-band, 16 GHz bandwidth
GEOMETRIES = {'Conventional': (conventional_aircraft, PALETTE['CONV']),
              'Stealth': (stealth_aircraft, PALETTE['STLTH'])}


def coherent_amplitude(solver, shat):
    """Full-physics coherent complex amplitude -- sums whichever layers the
    solver class provides (PO always; +edge/+double-bounce when present).
    Mirrors exactly what monostatic_rcs_dbsm/rcs_breakdown do internally,
    just without the final |.|^2 -> dB step, so phase is preserved for IFFT."""
    amp = solver._po_amplitude(shat)
    if hasattr(solver, '_edge_amplitude'):
        amp = amp + solver._edge_amplitude(shat)
    if hasattr(solver, '_double_bounce_amplitude'):
        amp = amp + solver._double_bounce_amplitude(shat)
    return amp


def frequency_response(builder, az_deg, freqs, subdiv=2, SolverCls=DoubleBounce):
    v, f = builder(subdiv=subdiv)
    H = np.empty(len(freqs), dtype=complex)
    for i, fr in enumerate(freqs):
        s = SolverCls(v, f, fr)
        shat = s._shat(az_deg, 0.0)
        H[i] = coherent_amplitude(s, shat)
    return H


def range_profile(H, freqs):
    """IFFT a uniformly-sampled frequency response into a range profile."""
    window = np.hanning(len(freqs))
    h = np.fft.ifftshift(np.fft.ifft(window * H))
    df = freqs[1] - freqs[0]
    dt = 1.0 / (len(freqs) * df)
    t = (np.arange(len(freqs)) - len(freqs) // 2) * dt
    R = C_LIGHT * t / 2.0
    return R, np.abs(h)


def freq_response_plot():
    freqs = np.linspace(F_LOW, F_HIGH, N_FREQ)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, (builder, color)) in zip(axes, GEOMETRIES.items()):
        t0 = time.time()
        H = frequency_response(builder, 0.0, freqs)
        rcs_db = 10.0 * np.log10((4 * np.pi / (C_LIGHT / freqs) ** 2) * np.abs(H) ** 2 + 1e-30)
        ax.plot(freqs / 1e9, rcs_db, color=color, lw=1.4)
        style_axis(ax, title=f'{name} — nose-on wideband RCS vs frequency ({time.time()-t0:.1f}s)',
                   xlabel='Frequency (GHz)', ylabel='RCS (dBsm)')
    fig.suptitle('Wideband frequency response — coherent PO+PTD+DB, 2-18 GHz',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/wideband_frequency_response.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/wideband_frequency_response.png')
    return freqs


def range_profile_plot(freqs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    dR = C_LIGHT / (2 * (F_HIGH - F_LOW))
    for ax, (name, (builder, color)) in zip(axes, GEOMETRIES.items()):
        H = frequency_response(builder, 0.0, freqs)
        R, mag = range_profile(H, freqs)
        mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-30) + 1e-6)
        ax.plot(R, mag_db, color=color, lw=1.6)
        ax.set_xlim(-6, 6)
        style_axis(ax, title=f'{name} — range profile (nose-on), ΔR={dR*100:.1f} cm resolution',
                   xlabel='Range (m, relative to target center)', ylabel='Normalized magnitude (dB)')
    fig.suptitle('Wideband impulse response / range profile — IFFT of 2-18 GHz coherent sweep',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/wideband_range_profile.png', dpi=150)
    plt.close(fig)
    print(f'Saved {OUT}/wideband_range_profile.png')


def convergence_study():
    """Mesh convergence of wideband RCS energy (mean |H(f)|^2 across the
    band, broadside where multiple scatterers interact most) -- fewer
    frequency points than the main sweep to stay within reasonable runtime
    across the O(n^2) DoubleBounce tier."""
    freqs = np.linspace(F_LOW, F_HIGH, 32)
    TIERS = ([(s, DoubleBounce) for s in [1, 2, 4, 8]] +
             [(s, PTDSolver) for s in [16, 32, 64]] +
             [(s, POAmplitudeSolver) for s in [150, 300, 600]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, (builder, color)) in zip(axes, GEOMETRIES.items()):
        facets, energies = [], []
        for subdiv, SolverCls in TIERS:
            t0 = time.time()
            H = frequency_response(builder, 90.0, freqs, subdiv=subdiv, SolverCls=SolverCls)
            v, f = builder(subdiv=subdiv)
            energy_db = 10.0 * np.log10(np.mean(np.abs(H) ** 2) + 1e-30)
            facets.append(len(f))
            energies.append(energy_db)
            print(f"{name:<14} subdiv={subdiv:<5} facets={len(f):<10,} {SolverCls.__name__:<18} "
                  f"band_energy={energy_db:7.2f} dB ({time.time()-t0:.1f}s)")
        ax.plot(facets, energies, marker='o', ms=6, lw=2.2, color=color)
        ax.set_xscale('log')
        style_axis(ax, title=f'{name} — wideband energy convergence (broadside)',
                   xlabel='Facet count (log scale)', ylabel='Mean band |H|² (dB)')
    fig.suptitle('Mesh convergence — wideband coherent response',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/wideband_mesh_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}/wideband_mesh_convergence.png')


if __name__ == '__main__':
    freqs = freq_response_plot()
    range_profile_plot(freqs)
    convergence_study()
    print('\nAll wideband impulse-response plots saved to', OUT)
