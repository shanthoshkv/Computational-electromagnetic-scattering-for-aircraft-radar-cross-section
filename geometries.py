"""
geometries.py
=============
Triangulated mesh definitions for:
  - Sphere
  - Flat Plate
  - Conventional Aircraft (box fuselage, flat wings, vertical stabilizer)
  - Stealth Aircraft (F-117/B-2 inspired: faceted, swept, canted)

All coordinates in metres. Aircraft nose points in -X direction.
Radar azimuth 0° = nose-on, 90° = right broadside.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _quad(p0, p1, p2, p3, subdiv=1):
    """Return (vertices, faces) for a planar quad, optionally subdivided
    into an subdiv x subdiv grid of sub-quads (2 triangles each) for mesh
    convergence studies. subdiv=1 reproduces the original 4-vert/2-face quad.
    Vectorised (np.meshgrid) — a Python double-loop here caps out around
    10^4 points; this scales to 10^7+ in seconds."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    n = subdiv
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    v_ = (ii.ravel() / n)[:, None]
    u_ = (jj.ravel() / n)[:, None]
    verts = ((1-u_)*(1-v_)*p0 + u_*(1-v_)*p1 + u_*v_*p2 + (1-u_)*v_*p3)

    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    a = (i_idx*(n+1) + j_idx).ravel()
    b = (i_idx*(n+1) + j_idx + 1).ravel()
    c = ((i_idx+1)*(n+1) + j_idx + 1).ravel()
    d = ((i_idx+1)*(n+1) + j_idx).ravel()
    faces = np.concatenate([np.stack([a, b, c], axis=1),
                             np.stack([a, c, d], axis=1)], axis=0)
    return verts, faces


def _weld(vertices, faces, tol=1e-6):
    """Merge coincident vertices so adjacent facets share indices — required
    for PTD edge classification (internal vs boundary) to see shared edges
    correctly after a primitive is built from independently-parametrised
    quads (e.g. a subdivided box's six faces). Vectorised via np.unique —
    a Python dict-loop here is the bottleneck past ~10^5 vertices."""
    keys = np.round(vertices / tol).astype(np.int64)
    _, idx_first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return vertices[idx_first], inverse[faces]


def _box(xmin, xmax, ymin, ymax, zmin, zmax, subdiv=1):
    """Return (vertices, faces) for a closed box, each face optionally
    subdivided. Faces are welded so box edges are detected as internal
    (shared) edges by the PTD solver, matching the original construction."""
    v0 = [xmin, ymin, zmin]; v1 = [xmax, ymin, zmin]
    v2 = [xmax, ymax, zmin]; v3 = [xmin, ymax, zmin]
    v4 = [xmin, ymin, zmax]; v5 = [xmax, ymin, zmax]
    v6 = [xmax, ymax, zmax]; v7 = [xmin, ymax, zmax]
    v, f = _combine([
        _quad(v0, v3, v2, v1, subdiv),   # -Z face
        _quad(v4, v5, v6, v7, subdiv),   # +Z face
        _quad(v0, v1, v5, v4, subdiv),   # -Y face
        _quad(v3, v7, v6, v2, subdiv),   # +Y face
        _quad(v1, v2, v6, v5, subdiv),   # +X face
        _quad(v0, v4, v7, v3, subdiv),   # -X face
    ])
    return _weld(v, f)


def _tri(p0, p1, p2):
    v = np.array([p0, p1, p2], dtype=float)
    f = np.array([[0, 1, 2]])
    return v, f


def _combine(mesh_list):
    """Merge a list of (vertices, faces) into one mesh."""
    all_v, all_f, offset = [], [], 0
    for v, f in mesh_list:
        all_v.append(np.asarray(v, dtype=float))
        all_f.append(np.asarray(f) + offset)
        offset += len(v)
    return np.vstack(all_v), np.vstack(all_f)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: SPHERE
# ─────────────────────────────────────────────────────────────────────────────

def sphere_mesh(radius=1.0, n_lat=24, n_lon=48):
    """UV-sphere triangulation."""
    verts = []
    for i in range(n_lat + 1):
        phi = -np.pi / 2 + np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2 * np.pi * j / n_lon
            verts.append([
                radius * np.cos(phi) * np.cos(theta),
                radius * np.cos(phi) * np.sin(theta),
                radius * np.sin(phi),
            ])
    verts = np.array(verts)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces += [[a, b, c], [a, c, d]]
    return verts, np.array(faces)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: FLAT PLATE
# ─────────────────────────────────────────────────────────────────────────────

def flat_plate(length=2.0, width=2.0):
    """Square flat plate in the XY plane (normal = +Z)."""
    l, w = length / 2, width / 2
    return _quad([-l, -w, 0], [l, -w, 0], [l, w, 0], [-l, w, 0])


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: CONVENTIONAL AIRCRAFT
# ─────────────────────────────────────────────────────────────────────────────
#
#  Design intent: LOTS of flat surfaces at 0°/90° angles.
#  Strong RCS contributors:
#    • Box fuselage sides  → broadside (90°)
#    • Flat wings          → top/bottom look angles
#    • Vertical stabilizer → broadside (90°) + corner dihedral with fuselage
#    • Flat nose face      → nose-on (0°)
#    • 90° wing-fuselage dihedral → corner reflector effect
#

def conventional_aircraft(subdiv=1):
    meshes = []

    # ── Fuselage: simple box, flat sides give strong broadside return ──
    meshes.append(_box(-3.6, 3.6, -0.50, 0.50, -0.50, 0.50, subdiv))

    # ── Main wings: flat plates, NO dihedral (0°) ──
    meshes.append(_quad(  # right wing
        [-1.2,  0.50, 0.0], [ 1.4,  0.50, 0.0],
        [ 1.4,  3.00, 0.0], [-1.2,  3.00, 0.0], subdiv))
    meshes.append(_quad(  # left wing
        [-1.2, -0.50, 0.0], [-1.2, -3.00, 0.0],
        [ 1.4, -3.00, 0.0], [ 1.4, -0.50, 0.0], subdiv))

    # ── Vertical stabilizer: perfectly vertical flat slab ──
    #    Creates a strong dihedral corner with horizontal fuselage top → classic RCS spike
    meshes.append(_box(1.6, 3.4, -0.12, 0.12, 0.50, 1.70, subdiv))

    # ── Horizontal stabilizer ──
    meshes.append(_quad(  # right
        [1.6,  0.50, -0.20], [3.4,  0.50, -0.20],
        [3.4,  1.40, -0.20], [1.6,  1.40, -0.20], subdiv))
    meshes.append(_quad(  # left
        [1.6, -0.50, -0.20], [1.6, -1.40, -0.20],
        [3.4, -1.40, -0.20], [3.4, -0.50, -0.20], subdiv))

    # ── Nose face (simulates flat engine intake) ──
    meshes.append(_quad(
        [-3.6, -0.50, -0.50], [-3.6,  0.50, -0.50],
        [-3.6,  0.50,  0.50], [-3.6, -0.50,  0.50], subdiv))

    return _combine(meshes)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: STEALTH AIRCRAFT
# ─────────────────────────────────────────────────────────────────────────────
#
#  Design intent: REDIRECT energy away from the threat axis (0° ± ~30°).
#  Key principles applied:
#    • All vertical surfaces canted (no 90° walls facing the radar)
#    • Fuselage has trapezoidal cross-section (angled sides)
#    • Delta wings with ~65° leading-edge sweep → leading edge spike off nose axis
#    • V-tail canted 35° outward, no perpendicular fin surface
#    • Pointed nose → tiny nose-on area
#    • No right-angle dihedrals → no corner-reflector effect
#

def stealth_aircraft(subdiv=1, sweep_deg=None, cant_deg=35.0, nose_length=2.0):
    """sweep_deg=None reproduces the exact baseline wing geometry bit-for-bit
    (used by the report/mesh-convergence studies). Pass a number to sweep
    the leading-edge angle for the geometry optimizer -- cant_deg and
    nose_length default to the baseline's implicit values (35 deg, 2.0 m)."""
    meshes = []
    c = np.radians(cant_deg)   # cant angle for V-tail

    # ── Fuselage: trapezoidal cross-section (wide bottom, narrower top) ──
    #   Top panel (flat, but tilted slightly)
    meshes.append(_quad(
        [-3.2, -0.38, 0.46], [-3.2,  0.38, 0.46],
        [ 3.0,  0.38, 0.46], [ 3.0, -0.38, 0.46], subdiv))
    #   Bottom panel
    meshes.append(_quad(
        [-3.2, -0.55, -0.12], [ 3.0, -0.55, -0.12],
        [ 3.0,  0.55, -0.12], [-3.2,  0.55, -0.12], subdiv))
    #   Right side (angled inward ~20°)
    meshes.append(_quad(
        [-3.2,  0.38, 0.46], [ 3.0,  0.38, 0.46],
        [ 3.0,  0.55, -0.12], [-3.2,  0.55, -0.12], subdiv))
    #   Left side (angled inward ~20°)
    meshes.append(_quad(
        [-3.2, -0.55, -0.12], [ 3.0, -0.55, -0.12],
        [ 3.0, -0.38,  0.46], [-3.2, -0.38,  0.46], subdiv))

    # ── Nose section: pointed, 4-facet pyramid → tiny frontal area ──
    tip = np.array([-3.2 - nose_length, 0.0, 0.17])
    base = [
        np.array([-3.2, -0.38, -0.12]),
        np.array([-3.2,  0.38, -0.12]),
        np.array([-3.2,  0.38,  0.46]),
        np.array([-3.2, -0.38,  0.46]),
    ]
    for i in range(4):
        meshes.append(_tri(base[i], base[(i+1) % 4], tip))

    # ── Highly swept delta wings ──
    #    Leading edge spike swept back from nose-on → non-threatening sector
    if sweep_deg is None:
        # baseline geometry (bit-exact, matches report/convergence-study numbers)
        meshes.append(_quad(  # right wing
            [-0.4,  0.55, -0.10], [ 2.6,  0.55, -0.10],
            [-1.8,  3.40, -0.10], [-3.2,  3.40, -0.10], subdiv))
        meshes.append(_quad(  # left wing
            [-0.4, -0.55, -0.10], [-3.2, -3.40, -0.10],
            [-1.8, -3.40, -0.10], [ 2.6, -0.55, -0.10], subdiv))
    else:
        span_y = 3.40 - 0.55   # same spanwise extent as baseline
        dx = span_y * np.tan(np.radians(sweep_deg))
        root_le_x, root_te_x = -0.4, 2.6
        tip_le_x, tip_te_x = root_le_x - dx, root_te_x - dx
        meshes.append(_quad(  # right wing
            [root_le_x,  0.55, -0.10], [root_te_x,  0.55, -0.10],
            [tip_te_x,  3.40, -0.10], [tip_le_x,  3.40, -0.10], subdiv))
        meshes.append(_quad(  # left wing
            [root_le_x, -0.55, -0.10], [tip_le_x, -3.40, -0.10],
            [tip_te_x, -3.40, -0.10], [root_te_x, -0.55, -0.10], subdiv))

    # ── Canted V-tails: angled 35° outward, no vertical wall ──
    meshes.append(_quad(  # right V-tail
        [ 2.0,  0.38,  0.46],
        [ 3.2,  0.38,  0.46],
        [ 3.2,  0.38 + 0.95*np.sin(c),  0.46 + 0.95*np.cos(c)],
        [ 2.0,  0.38 + 0.65*np.sin(c),  0.46 + 0.65*np.cos(c)], subdiv))
    meshes.append(_quad(  # left V-tail
        [ 2.0, -0.38,  0.46],
        [ 2.0, -0.38 - 0.65*np.sin(c),  0.46 + 0.65*np.cos(c)],
        [ 3.2, -0.38 - 0.95*np.sin(c),  0.46 + 0.95*np.cos(c)],
        [ 3.2, -0.38,  0.46], subdiv))

    return _combine(meshes)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

GEOMETRIES = {
    "Sphere (r=1m)":             lambda: sphere_mesh(radius=1.0),
    "Flat Plate (2×2 m)":        lambda: flat_plate(2.0, 2.0),
    "Conventional Aircraft":     conventional_aircraft,
    "Stealth Aircraft":          stealth_aircraft,
    "Flying Wing (B-2 style)":   lambda: flying_wing(),
    "Chined Nose (F-22 style)":  lambda: chined_nose_aircraft(),
}


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: FLYING WING (B-2 style)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Design intent: NO vertical surfaces at all, NO fuselage/wing distinction.
#  Single continuous lifting surface, double-W trailing edge for edge
#  alignment, sawtooth panel-to-panel joints.
#

def flying_wing(span=12.0, root_chord=6.0, tip_chord=1.2, sweep_deg=33.0):
    meshes = []
    sweep = np.radians(sweep_deg)
    half_span = span / 2.0
    le_sweep_offset = half_span * np.tan(sweep)

    # ── Top surface: single faceted "kite" shape, slight camber via 2 panels ──
    # Centre section
    apex   = np.array([-root_chord*0.55,  0.0,  0.30])     # nose apex
    le_l   = np.array([apex[0] + le_sweep_offset, -half_span, 0.05])
    le_r   = np.array([apex[0] + le_sweep_offset,  half_span, 0.05])
    te_l   = np.array([apex[0] + root_chord*0.9 - half_span*0.25, -half_span*0.55, 0.05])
    te_r   = np.array([apex[0] + root_chord*0.9 - half_span*0.25,  half_span*0.55, 0.05])
    te_tip_l = np.array([apex[0] + root_chord*0.55, -half_span, 0.05])
    te_tip_r = np.array([apex[0] + root_chord*0.55,  half_span, 0.05])

    # Top skin (slightly cambered: apex raised in z relative to edges)
    meshes.append(_tri(apex, le_l, le_r))                # nose top facet
    meshes.append(_quad(le_l, le_r, te_r, te_l))         # centre top facet
    meshes.append(_tri(le_l, te_l, te_tip_l))            # left wingtip top
    meshes.append(_tri(le_r, te_tip_r, te_r))            # right wingtip top

    # Bottom skin (flat, z = -0.12)
    apex_b   = apex.copy();   apex_b[2]   = -0.12
    le_l_b   = le_l.copy();   le_l_b[2]   = -0.12
    le_r_b   = le_r.copy();   le_r_b[2]   = -0.12
    te_l_b   = te_l.copy();   te_l_b[2]   = -0.12
    te_r_b   = te_r.copy();   te_r_b[2]   = -0.12
    te_tip_l_b = te_tip_l.copy(); te_tip_l_b[2] = -0.12
    te_tip_r_b = te_tip_r.copy(); te_tip_r_b[2] = -0.12

    meshes.append(_tri(apex_b, le_r_b, le_l_b))
    meshes.append(_quad(le_l_b, te_l_b, te_r_b, le_r_b))
    meshes.append(_tri(le_l_b, te_tip_l_b, te_l_b))
    meshes.append(_tri(le_r_b, te_r_b, te_tip_r_b))

    # Leading-edge skin (thin connecting strip, gives the body thickness)
    meshes.append(_quad(apex, apex_b, le_l_b, le_l))
    meshes.append(_quad(apex, le_r, le_r_b, apex_b))
    meshes.append(_quad(le_l, le_l_b, te_tip_l_b, te_tip_l))
    meshes.append(_quad(le_r, te_tip_r, te_tip_r_b, le_r_b))

    # Sawtooth trailing edge (double-W): 4 small zigzag facets per side
    def sawtooth(p_in, p_out, p_in_b, p_out_b, n_teeth=2):
        out = []
        span_vec = p_out - p_in
        for i in range(n_teeth):
            t0, t1 = i / n_teeth, (i + 0.5) / n_teeth
            t2 = (i + 1) / n_teeth
            a  = p_in + t0 * span_vec
            b  = p_in + t1 * span_vec + np.array([0.4, 0.0, 0.0])  # tooth tip
            c  = p_in + t2 * span_vec
            a_b, b_b, c_b = a.copy(), b.copy(), c.copy()
            a_b[2] = b_b[2] = c_b[2] = -0.12
            out.append(_tri(a, b, c))
            out.append(_tri(a_b, c_b, b_b))
        return out

    meshes += sawtooth(te_l, te_tip_l, te_l_b, te_tip_l_b)
    meshes += sawtooth(te_tip_r, te_r, te_tip_r_b, te_r_b)
    meshes.append(_quad(te_l, te_r, te_r_b, te_l_b))  # centre trailing edge

    return _combine(meshes)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: CHINED-NOSE AIRCRAFT (F-22 style)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Design intent: forebody chines blend fuselage into wing for edge alignment;
#  diamond cross-section; canted twin tails; serrated panel edges.
#

def chined_nose_aircraft():
    meshes = []
    c = np.radians(28)  # tail cant angle

    # ── Forebody: diamond cross-section narrowing to chine edges ──
    nose_tip = np.array([-5.0, 0.0, 0.10])
    chine_l  = np.array([-2.0, -1.0, 0.0])
    chine_r  = np.array([-2.0,  1.0, 0.0])
    top_mid  = np.array([-2.0,  0.0, 0.55])
    bot_mid  = np.array([-2.0,  0.0, -0.35])

    # Forebody facets (4 facets fanning from nose tip to diamond cross-section)
    meshes.append(_tri(nose_tip, chine_l, top_mid))
    meshes.append(_tri(nose_tip, top_mid, chine_r))
    meshes.append(_tri(nose_tip, chine_r, bot_mid))
    meshes.append(_tri(nose_tip, bot_mid, chine_l))

    # ── Mid-fuselage / wing-blended section (diamond extruded back) ──
    top_mid2 = np.array([2.5,  0.0,  0.55])
    bot_mid2 = np.array([2.5,  0.0, -0.35])
    chine_l2 = np.array([2.5, -2.6,  0.0])
    chine_r2 = np.array([2.5,  2.6,  0.0])

    meshes.append(_quad(top_mid, chine_l, chine_l2, top_mid2))   # left-top blended panel
    meshes.append(_quad(top_mid, top_mid2, chine_r2, chine_r))   # right-top blended panel
    meshes.append(_quad(bot_mid, bot_mid2, chine_l2, chine_l))   # left-bottom panel
    meshes.append(_quad(bot_mid, chine_r, chine_r2, bot_mid2))   # right-bottom panel

    # ── Aft fuselage closure (diamond cross-section, flat back) ──
    meshes.append(_tri(top_mid2, chine_l2, bot_mid2))
    meshes.append(_tri(top_mid2, bot_mid2, chine_r2))

    # ── Canted twin vertical tails (28° outward, no 90° dihedral) ──
    for side, y0 in [(+1, 0.9), (-1, -0.9)]:
        base_f = np.array([1.0, y0, 0.30])
        base_a = np.array([2.4, y0, 0.30])
        tip_f  = np.array([1.3, y0 + side*1.4*np.sin(c), 0.30 + 1.4*np.cos(c)])
        tip_a  = np.array([2.4, y0 + side*1.1*np.sin(c), 0.30 + 1.1*np.cos(c)])
        meshes.append(_quad(base_f, base_a, tip_a, tip_f))

    return _combine(meshes)

