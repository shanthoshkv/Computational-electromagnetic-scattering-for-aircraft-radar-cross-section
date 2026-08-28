"""
main.py
=======
Generates a LARGE set of individual, single-subject PNG images suitable
for direct insertion into PowerPoint slides. Every image is one self-
contained figure — no multi-panel grids — so each can be dropped onto
a slide and resized freely.

Output structure (all in ./output_ppt/):

  01_geometry/
      <name>_iso.png, <name>_front.png, <name>_side.png, <name>_top.png
      <name>_wireframe.png
      (for all 6 geometries)

  02_rcs_polar/
      <name>_polar_PO.png
      <name>_polar_full.png        (PO+PTD+DoubleBounce)
      (for conventional + stealth + flying wing + chined nose)

  03_rcs_heatmap/
      <name>_heatmap.png           (az x el contour, full physics)

  04_frequency/
      <name>_freq_nose.png
      <name>_freq_broadside.png

  05_ram/
      ram_absorption_curves.png
      stealth_with_<material>.png

  06_breakdown/
      <name>_mechanism_<mech>.png  (PO / PTD / DoubleBounce individually)

  07_comparison/
      overlay_polar_all6.png
      ranking_bar_chart.png
      reduction_bar_chart.png

  08_physics_diagrams/
      salisbury_diagram.png
      dallenbach_diagram.png
      <name>_edge_classification.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D                # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from geometries import (conventional_aircraft, stealth_aircraft, sphere_mesh,
                         flat_plate, flying_wing, chined_nose_aircraft)
from solvers import (POAmplitudeSolver, PTDSolver, DoubleBounce,
                      SalisburyScreen, DallenbachLayer,
                      physics_frequency_sweep, breakdown_sweep, mie_rcs_dbsm)

BASE = './output_ppt'
DIRS = ['01_geometry', '02_rcs_polar', '03_rcs_heatmap', '04_frequency',
        '05_ram', '06_breakdown', '07_comparison', '08_physics_diagrams']
for d in DIRS:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────
BG, PANEL, GRID_C = '#080C14', '#0E1420', '#1E2A3A'
TEXT, DIM = '#D0DCF0', '#6A7A9A'
C_CONV, C_STLTH = '#FF5B35', '#00E5FF'
C_SPHERE, C_PLATE = '#FFD040', '#7CFC00'
C_WING, C_CHINE = '#FF6EC7', '#39FF88'
C_PO, C_PTD, C_DB, C_FULL = '#888888', '#FFD040', '#C084FC', '#00FF88'

RADAR_CMAP = LinearSegmentedColormap.from_list('radar',
    ['#080C14','#0A2040','#004080','#0080C0','#00C0E0',
     '#40E080','#C0E040','#FFB000','#FF4000'])
EDGE_CMAP = LinearSegmentedColormap.from_list('edge',
    ['#1A2A1A','#00AA44','#FFD040','#FF4400'])

FREQ_HZ, FREQ_GHZ = 10e9, 10.0
FREQS = np.linspace(1e9, 36e9, 90)
FREQS_G = FREQS / 1e9

def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.xaxis.label.set_color(DIM); ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_color(DIM); ax.yaxis.label.set_size(10)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(color=GRID_C, linewidth=0.6, linestyle='--', alpha=0.8)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_C)

def style_polar(ax, title='', r_min=-40, r_max=40):
    ax.set_facecolor(PANEL)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_ylim(r_min, r_max)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=14)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(color=GRID_C, linewidth=0.5, alpha=0.7)
    ax.spines['polar'].set_color(GRID_C)
    ticks = np.arange(r_min, r_max+1, 10)
    ax.set_yticks(ticks); ax.set_yticklabels([f'{t}' for t in ticks], color=DIM, fontsize=7)
    ax.set_xticklabels(['N\n0°','45°','E\n90°','135°','S\n180°','225°','W\n270°','315°'],
                        fontsize=8, color=DIM)

def draw_geom_3d(ax, V, F, color, alpha=0.75, lw=0.2, wireframe=False):
    if wireframe:
        edges_seen = set()
        segs = []
        for f in F:
            for k in range(3):
                a, b = int(f[k]), int(f[(k+1)%3])
                key = (min(a,b), max(a,b))
                if key not in edges_seen:
                    edges_seen.add(key)
                    segs.append([V[a], V[b]])
        lc = Line3DCollection(segs, colors=color, linewidths=0.8, alpha=0.9)
        ax.add_collection3d(lc)
    else:
        tris = [V[f] for f in F]
        poly = Poly3DCollection(tris, alpha=alpha, linewidth=lw,
                                 facecolor=color, edgecolor='#2A3A50')
        ax.add_collection3d(poly)
    margin = 0.3
    ax.set_xlim(V[:,0].min()-margin, V[:,0].max()+margin)
    ax.set_ylim(V[:,1].min()-margin, V[:,1].max()+margin)
    ax.set_zlim(V[:,2].min()-margin, V[:,2].max()+margin)
    ax.set_facecolor(BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#1A2434')
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color(DIM); axis.label.set_size(8)
    ax.tick_params(colors='#3A4A60', labelsize=7)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

def save_single(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  → {path.replace(BASE+'/', '')}")


print("Building geometries …")
GEOMS = {
    'sphere':      (sphere_mesh(radius=1.0),      C_SPHERE),
    'flat_plate':  (flat_plate(2.0, 2.0),          C_PLATE),
    'conventional':(conventional_aircraft(),       C_CONV),
    'stealth':     (stealth_aircraft(),             C_STLTH),
    'flying_wing': (flying_wing(),                  C_WING),
    'chined_nose': (chined_nose_aircraft(),         C_CHINE),
}
for name, ((V, F), color) in GEOMS.items():
    print(f"  {name:14s}: {len(F):4d} facets")


# ═════════════════════════════════════════════════════════════════════════════
# 01 — INDIVIDUAL GEOMETRY VIEWS  (iso / front / side / top / wireframe)
# ═════════════════════════════════════════════════════════════════════════════

print("\n[01] Individual geometry views …")

VIEW_ANGLES = {
    'iso':   (25, -135),
    'front': (5,  -90),
    'side':  (5,    0),
    'top':   (89, -90),
}

for name, ((V, F), color) in GEOMS.items():
    for view_name, (elev, azim) in VIEW_ANGLES.items():
        fig = plt.figure(figsize=(8, 7), facecolor=BG)
        ax = fig.add_subplot(111, projection='3d')
        draw_geom_3d(ax, V, F, color)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name.replace('_',' ').title()} — {view_name.title()} View",
                      color=color, fontsize=13, fontweight='bold', pad=10)
        save_single(fig, f'{BASE}/01_geometry/{name}_{view_name}.png')

    # Wireframe (isometric)
    fig = plt.figure(figsize=(8, 7), facecolor=BG)
    ax = fig.add_subplot(111, projection='3d')
    draw_geom_3d(ax, V, F, color, wireframe=True)
    ax.view_init(elev=25, azim=-135)
    ax.set_title(f"{name.replace('_',' ').title()} — Wireframe / Mesh Structure",
                  color=color, fontsize=13, fontweight='bold', pad=10)
    save_single(fig, f'{BASE}/01_geometry/{name}_wireframe.png')


# ═════════════════════════════════════════════════════════════════════════════
# 02 — INDIVIDUAL RCS POLAR PLOTS  (PO-only vs Full-physics, per aircraft)
# ═════════════════════════════════════════════════════════════════════════════

print("\n[02] Individual RCS polar plots …")

N_AZ = 180
az_angles = np.linspace(0, 360, N_AZ, endpoint=False)
az_rad = np.radians(az_angles)

aircraft_for_polar = [('conventional', C_CONV), ('stealth', C_STLTH),
                       ('flying_wing', C_WING), ('chined_nose', C_CHINE)]

for name, color in aircraft_for_polar:
    (V, F), _ = GEOMS[name]
    print(f"  {name}: PO-only sweep …")
    s_po = POAmplitudeSolver(V, F, FREQ_HZ)
    rcs_po = np.array([s_po.monostatic_rcs_dbsm(a) for a in az_angles])

    print(f"  {name}: Full-physics sweep …")
    s_full = DoubleBounce(V, F, FREQ_HZ)
    rcs_full = np.array([s_full.monostatic_rcs_dbsm(a) for a in az_angles])

    r_min, r_max = min(rcs_po.min(), rcs_full.min())-3, max(rcs_po.max(), rcs_full.max())+3

    # PO-only individual plot
    fig = plt.figure(figsize=(8, 8), facecolor=BG)
    ax = fig.add_subplot(111, projection='polar')
    style_polar(ax, f"{name.replace('_',' ').title()} — RCS (PO only)\nX-band 10 GHz",
                r_min=r_min, r_max=r_max)
    ax.plot(az_rad, rcs_po, color=color, lw=2.0)
    ax.fill(az_rad, rcs_po, color=color, alpha=0.15)
    save_single(fig, f'{BASE}/02_rcs_polar/{name}_polar_PO.png')

    # Full physics individual plot
    fig = plt.figure(figsize=(8, 8), facecolor=BG)
    ax = fig.add_subplot(111, projection='polar')
    style_polar(ax, f"{name.replace('_',' ').title()} — RCS (Full Physics)\nPO+PTD+DoubleBounce, X-band 10 GHz",
                r_min=r_min, r_max=r_max)
    ax.plot(az_rad, rcs_full, color=color, lw=2.0)
    ax.fill(az_rad, rcs_full, color=color, alpha=0.15)
    save_single(fig, f'{BASE}/02_rcs_polar/{name}_polar_full.png')

# Sphere (Mie, analytic) and flat plate (PO)
fig = plt.figure(figsize=(8, 8), facecolor=BG)
ax = fig.add_subplot(111, projection='polar')
sphere_val = mie_rcs_dbsm(1.0, FREQ_HZ)
style_polar(ax, "Sphere (r=1m) — RCS (Mie Series, exact)\nX-band 10 GHz",
            r_min=sphere_val-10, r_max=sphere_val+10)
ax.plot(az_rad, np.full(N_AZ, sphere_val), color=C_SPHERE, lw=2.5)
ax.fill(az_rad, np.full(N_AZ, sphere_val), color=C_SPHERE, alpha=0.15)
save_single(fig, f'{BASE}/02_rcs_polar/sphere_polar_mie.png')

(V_p, F_p), _ = GEOMS['flat_plate']
s_plate = POAmplitudeSolver(V_p, F_p, FREQ_HZ)
rcs_plate = np.array([s_plate.monostatic_rcs_dbsm(a) for a in az_angles])
fig = plt.figure(figsize=(8, 8), facecolor=BG)
ax = fig.add_subplot(111, projection='polar')
style_polar(ax, "Flat Plate (2×2m) — RCS (PO)\nX-band 10 GHz",
            r_min=rcs_plate.min()-5, r_max=rcs_plate.max()+5)
ax.plot(az_rad, rcs_plate, color=C_PLATE, lw=2.0)
ax.fill(az_rad, rcs_plate, color=C_PLATE, alpha=0.15)
save_single(fig, f'{BASE}/02_rcs_polar/flat_plate_polar_PO.png')


# ═════════════════════════════════════════════════════════════════════════════
# 03 — INDIVIDUAL RCS HEATMAPS (az × el, full physics)
# ═════════════════════════════════════════════════════════════════════════════

print("\n[03] Individual RCS heatmaps …")

for name, color in [('conventional', C_CONV), ('stealth', C_STLTH)]:
    (V, F), _ = GEOMS[name]
    print(f"  {name}: heatmap (this is the slow one) …")
    s = DoubleBounce(V, F, FREQ_HZ)
    azs  = np.linspace(0, 360, 90, endpoint=False)
    els  = np.linspace(-75, 75, 45)
    grid = np.empty((len(els), len(azs)))
    for i, el in enumerate(els):
        for j, az in enumerate(azs):
            grid[i, j] = s.monostatic_rcs_dbsm(az, el)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    extent = [azs[0], azs[-1], els[0], els[-1]]
    im = ax.imshow(grid, aspect='auto', extent=extent, origin='lower', cmap=RADAR_CMAP)
    style_ax(ax, 'Azimuth (°)', 'Elevation (°)',
              f"{name.replace('_',' ').title()} — Full RCS Heatmap (PO+PTD+DB)\nX-band 10 GHz")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('RCS (dBsm)', color=TEXT, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM)
    for av in [0, 90, 180, 270]:
        ax.axvline(av, color='#FF8C00', lw=0.8, ls='--', alpha=0.5)
    save_single(fig, f'{BASE}/03_rcs_heatmap/{name}_heatmap.png')


# ═════════════════════════════════════════════════════════════════════════════
# 04 — INDIVIDUAL FREQUENCY SWEEP PLOTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n[04] Individual frequency sweep plots …")

for name, color in [('conventional', C_CONV), ('stealth', C_STLTH)]:
    (V, F), _ = GEOMS[name]
    print(f"  {name}: nose-on sweep …")
    rcs_nose = physics_frequency_sweep(V, F, FREQS, 0.0, level='full')
    print(f"  {name}: broadside sweep …")
    rcs_broad = physics_frequency_sweep(V, F, FREQS, 90.0, level='full')

    for angle_name, rcs_arr in [('nose', rcs_nose), ('broadside', rcs_broad)]:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
        style_ax(ax, 'Frequency (GHz)', 'RCS (dBsm)',
                  f"{name.replace('_',' ').title()} — RCS vs Frequency\n"
                  f"{'Nose-on (0°)' if angle_name=='nose' else 'Broadside (90°)'}, Full Physics")
        ax.plot(FREQS_G, rcs_arr, color=color, lw=2.2)
        ax.fill_between(FREQS_G, rcs_arr, rcs_arr.min()-3, color=color, alpha=0.12)
        ax.set_xlim(1, 36)
        for f0, f1, lbl in [(1,2,'L'),(2,4,'S'),(4,8,'C'),(8,12,'X'),(12,18,'Ku'),(18,27,'Ka')]:
            ax.axvspan(f0, f1, color='#1A2E44', alpha=0.3)
            ax.text((f0+f1)/2, ax.get_ylim()[0]+1, lbl, color='#4A6080', fontsize=8, ha='center')
        save_single(fig, f'{BASE}/04_frequency/{name}_freq_{angle_name}.png')


# ═════════════════════════════════════════════════════════════════════════════
# 05 — RAM MATERIAL PLOTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n[05] RAM material plots …")

ram_models = {
    'Salisbury f0=10GHz': SalisburyScreen(10e9),
    'Dallenbach Carbon Foam':   DallenbachLayer.from_preset('carbon_foam', 0.005),
    'Dallenbach Carbonyl Iron': DallenbachLayer.from_preset('carbonyl_iron', 0.005),
    'Dallenbach Ferrite Tile':  DallenbachLayer.from_preset('ferrite_tile', 0.005),
}
ram_colors = ['#FFD040', '#7CFC00', '#00E5FF', '#C084FC']

fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG)
style_ax(ax, 'Frequency (GHz)', 'One-way absorption (dB)',
          'RAM Material Absorption vs Frequency')
for (lbl, model), color in zip(ram_models.items(), ram_colors):
    ax.plot(FREQS_G, model.absorption_db(FREQS), color=color, lw=2.2, label=lbl)
ax.axvspan(8, 12, color='#1A2E44', alpha=0.4)
ax.text(10, ax.get_ylim()[0]+2, 'X-band', color='#4A6080', fontsize=9, ha='center', fontweight='bold')
ax.set_xlim(1, 36)
ax.legend(fontsize=9, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C)
save_single(fig, f'{BASE}/05_ram/ram_absorption_curves.png')

# Stealth aircraft with each RAM individually
(V_s, F_s), _ = GEOMS['stealth']
rcs_bare = physics_frequency_sweep(V_s, F_s, FREQS, 0.0, level='full')
for (lbl, model), color in zip(ram_models.items(), ram_colors):
    rcs_ram = physics_frequency_sweep(V_s, F_s, FREQS, 0.0, level='full', ram_model=model)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    style_ax(ax, 'Frequency (GHz)', 'RCS (dBsm)',
              f"Stealth Aircraft + {lbl}\nNose-on, Full Physics")
    ax.plot(FREQS_G, rcs_bare, color='white', lw=1.6, ls='--', label='Bare metal (shape only)')
    ax.plot(FREQS_G, rcs_ram, color=color, lw=2.2, label=f'+ {lbl}')
    ax.set_xlim(1, 36)
    ax.legend(fontsize=9, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C)
    fname = lbl.lower().replace(' ', '_').replace('=', '')
    save_single(fig, f'{BASE}/05_ram/stealth_with_{fname}.png')


# ═════════════════════════════════════════════════════════════════════════════
# 06 — INDIVIDUAL MECHANISM BREAKDOWN PLOTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n[06] Mechanism breakdown plots …")

mech_keys = [('PO only', C_PO, 'Specular Reflection (PO)'),
             ('PTD edge', C_PTD, 'Edge Diffraction (PTD)'),
             ('Double bounce', C_DB, 'Corner-Reflector Double Bounce')]

for name, color in [('conventional', C_CONV), ('stealth', C_STLTH)]:
    (V, F), _ = GEOMS[name]
    print(f"  {name}: breakdown sweep …")
    bd = breakdown_sweep(V, F, FREQ_HZ, n=180)

    for key, mcolor, full_label in mech_keys:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
        style_ax(ax, 'Azimuth (°)', 'RCS (dBsm)',
                  f"{name.replace('_',' ').title()} — {full_label}")
        ax.plot(bd['angles'], bd[key], color=mcolor, lw=2.0)
        ax.fill_between(bd['angles'], bd[key], bd[key].min()-5, color=mcolor, alpha=0.15)
        ax.plot(bd['angles'], bd['Full (PO+PTD+DB)'], color='white',
                lw=0.8, ls=':', alpha=0.4, label='Full total (reference)')
        ax.set_xlim(0, 360)
        ax.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C)
        fname = key.lower().replace(' ', '_')
        save_single(fig, f'{BASE}/06_breakdown/{name}_mechanism_{fname}.png')


# ═════════════════════════════════════════════════════════════════════════════
# 07 — COMPARISON PLOTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n[07] Comparison plots …")

# Overlay polar — all 6 geometries
all_rcs = {}
for name, ((V, F), color) in GEOMS.items():
    if name == 'sphere':
        all_rcs[name] = np.full(N_AZ, mie_rcs_dbsm(1.0, FREQ_HZ))
        continue
    s = POAmplitudeSolver(V, F, FREQ_HZ)
    all_rcs[name] = np.array([s.monostatic_rcs_dbsm(a) for a in az_angles])

r_min = min(r.min() for r in all_rcs.values()) - 3
r_max = max(r.max() for r in all_rcs.values()) + 3
fig = plt.figure(figsize=(10, 10), facecolor=BG)
ax = fig.add_subplot(111, projection='polar')
style_polar(ax, "All Geometries — RCS Comparison (PO)\nX-band 10 GHz", r_min=r_min, r_max=r_max)
for name, ((V, F), color) in GEOMS.items():
    ax.plot(az_rad, all_rcs[name], color=color, lw=1.8, alpha=0.9,
             label=name.replace('_', ' ').title())
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9,
           labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C)
save_single(fig, f'{BASE}/07_comparison/overlay_polar_all6.png')

# Ranking bar chart
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG)
style_ax(ax, '', 'Mean RCS (dBsm)', 'Stealth Ranking — Mean RCS Across All Azimuth Angles')
names_sorted = sorted(all_rcs.keys(), key=lambda n: all_rcs[n].mean())
means = [all_rcs[n].mean() for n in names_sorted]
colors_sorted = [GEOMS[n][1] for n in names_sorted]
bars = ax.bar([n.replace('_','\n') for n in names_sorted], means, color=colors_sorted, alpha=0.85)
for bar, val in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.1f}', ha='center',
             color=TEXT, fontsize=10, fontweight='bold')
save_single(fig, f'{BASE}/07_comparison/ranking_bar_chart.png')

# Reduction bar chart (conventional vs stealth at 4 angles)
(V_c, F_c), _ = GEOMS['conventional']
(V_st, F_st), _ = GEOMS['stealth']
s_c = DoubleBounce(V_c, F_c, FREQ_HZ)
s_st = DoubleBounce(V_st, F_st, FREQ_HZ)
angles_lbl = ['Nose-on\n0°', 'Quarter\n45°', 'Broadside\n90°', 'Tail-on\n180°']
angles_val = [0, 45, 90, 180]
deltas = [s_c.monostatic_rcs_dbsm(a) - s_st.monostatic_rcs_dbsm(a) for a in angles_val]
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
style_ax(ax, 'Look Angle', 'RCS Reduction ΔdB', 'RCS Reduction: Conventional → Stealth\n(Full Physics: PO+PTD+DoubleBounce)')
bar_colors = ['#00CC44' if d > 0 else '#FF4444' for d in deltas]
bars = ax.bar(angles_lbl, deltas, color=bar_colors, alpha=0.85, width=0.55)
for bar, val in zip(bars, deltas):
    ax.text(bar.get_x()+bar.get_width()/2, max(val,0)+0.5, f'{val:+.1f} dB',
             ha='center', color=TEXT, fontsize=10, fontweight='bold')
ax.axhline(0, color=DIM, lw=1, ls='--')
save_single(fig, f'{BASE}/07_comparison/reduction_bar_chart.png')


# ═════════════════════════════════════════════════════════════════════════════
# 08 — PHYSICS EXPLANATION DIAGRAMS
# ═════════════════════════════════════════════════════════════════════════════

print("\n[08] Physics diagrams …")

# Salisbury screen
fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
ax.set_facecolor(PANEL); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title("Salisbury Screen — Narrowband Absorber", color=TEXT, fontsize=13, fontweight='bold', pad=10)
for y0, y1, fc, label in [(8.5,9.3,'#3A6A9A','Resistive sheet  Rₛ = Z₀ = 377 Ω/□'),
                            (5.0,8.0,'#1A2E44','Lossless spacer  d = λ₀/4'),
                            (2.5,4.5,'#444','PEC ground plane')]:
    ax.fill_between([1,9],[y0,y0],[y1,y1], color=fc, alpha=0.75)
    ax.text(5, (y0+y1)/2, label, ha='center', va='center', color=TEXT, fontsize=10, fontweight='bold')
ax.annotate('', xy=(3,8.5), xytext=(1.3,9.8), arrowprops=dict(color='#3B8BD4', lw=2, arrowstyle='->'))
ax.text(1.0, 9.9, 'Eᵢ', color='#3B8BD4', fontsize=12)
ax.text(5, 1.2, 'Eᵣ → 0 when d=λ₀/4 and Rₛ=Z₀\nΓ(f) = (Z_in−Z₀)/(Z_in+Z₀)',
         ha='center', color=DIM, fontsize=10)
save_single(fig, f'{BASE}/08_physics_diagrams/salisbury_diagram.png')

# Dallenbach layer
fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
ax.set_facecolor(PANEL); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title("Dallenbach Layer — Broadband Absorber", color=TEXT, fontsize=13, fontweight='bold', pad=10)
ax.fill_between([1,9],[5,5],[8.5,8.5], color='#2A4A2A', alpha=0.75)
ax.text(5, 6.75, 'Lossy slab\nεᵣ = ε′−jε″,  μᵣ = μ′−jμ″\nthickness d',
         ha='center', va='center', color=TEXT, fontsize=10, fontweight='bold')
ax.fill_between([1,9],[3.5,3.5],[4.8,4.8], color='#444', alpha=0.8)
ax.text(5, 4.15, 'PEC ground plane', ha='center', va='center', color=TEXT, fontsize=10, fontweight='bold')
ax.text(5, 2.3, 'γ = j(2πf/c)√(μᵣεᵣ)', ha='center', color=C_PTD, fontsize=10)
ax.text(5, 1.5, 'Z_in = Z_m tanh(γd)', ha='center', color=C_PTD, fontsize=10)
ax.text(5, 0.7, 'Γ = (Z_in−Z₀)/(Z_in+Z₀)', ha='center', color=C_FULL, fontsize=10)
ax.annotate('', xy=(3,8.5), xytext=(1.3,9.6), arrowprops=dict(color='#3B8BD4', lw=2, arrowstyle='->'))
ax.text(1.0, 9.7, 'Eᵢ', color='#3B8BD4', fontsize=12)
save_single(fig, f'{BASE}/08_physics_diagrams/dallenbach_diagram.png')

# Edge wedge classification (n_wedge histograms, individual per aircraft)
for name, color in [('conventional', C_CONV), ('stealth', C_STLTH)]:
    (V, F), _ = GEOMS[name]
    s = PTDSolver(V, F, FREQ_HZ)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    style_ax(ax, 'Wedge parameter n', 'Edge count',
              f"{name.replace('_',' ').title()} — Edge Wedge-Angle Distribution\n"
              f"({len(s.edges)} total edges)")
    ax.hist(s.edge_n, bins=15, color=color, alpha=0.75, edgecolor=GRID_C)
    ax.axvline(1.5, color=C_DB, lw=1.8, ls='--', label='90° convex ridge (n=1.5)')
    ax.axvline(2.0, color=C_PTD, lw=1.8, ls='--', label='Half-plane free edge (n=2.0)')
    ax.axvline(2.5, color='#FF4444', lw=1.8, ls='--', label='Concave corner (n=2.5)')
    ax.legend(fontsize=9, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID_C)
    save_single(fig, f'{BASE}/08_physics_diagrams/{name}_edge_classification.png')


print("\n" + "="*70)
print(f"✓ ALL PPT ASSETS GENERATED in {BASE}")
total_files = sum(len(os.listdir(f'{BASE}/{d}')) for d in DIRS)
print(f"  Total images: {total_files}")
for d in DIRS:
    n = len(os.listdir(f'{BASE}/{d}'))
    print(f"    {d:25s}: {n:3d} files")
print("="*70)
