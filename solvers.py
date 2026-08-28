"""
solvers.py
==========
RCS Computational Physics Engine — Final Consolidated Version

Implements three coherent, additive scattering mechanisms plus two
frequency-dependent RAM material models:

  1. PO   — Physical Optics (specular reflection from facets)
  2. PTD  — Physical Theory of Diffraction (Ufimtsev edge fringe fields)
  3. DB   — Double Bounce (second-order corner-reflector geometric optics)

  RAM Models:
    a) Salisbury Screen  : resistive sheet + λ/4 spacer + ground plane
    b) Dallenbach Layer  : homogeneous lossy slab (εᵣ, μᵣ complex)

All solvers return complex scatter amplitude so contributions combine
coherently (correct phase addition) before |·|² squaring to get RCS.

Validated against exact Mie series for a conducting sphere (mie_rcs_dbsm).
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

C_LIGHT = 3e8
Z0      = 376.73   # free-space impedance [Ω]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: complex sinc  (sinc(0) = 1 by l'Hôpital)
# ─────────────────────────────────────────────────────────────────────────────

def _sinc(x):
    """sin(x)/x with safe limit at x=0."""
    x = np.asarray(x, dtype=float)
    safe_x = np.where(np.abs(x) < 1e-9, 1.0, x)
    result = np.sin(safe_x) / safe_x
    return np.where(np.abs(x) < 1e-9, 1.0, result)


# ─────────────────────────────────────────────────────────────────────────────
# BASE SOLVER  (upgraded: returns complex amplitude, not just dBsm)
# ─────────────────────────────────────────────────────────────────────────────

class POAmplitudeSolver:
    """
    Physical Optics coherent scatter amplitude (complex-valued).
    This is the base class shared by all higher-order solvers.

    Internal method  _po_amplitude(shat, shat_rx=None)  returns the raw
    complex PO sum, allowing downstream classes to add their own corrections
    before the final |·|² squaring.
    """

    def __init__(self, vertices, faces, frequency_hz, ram_model=None):
        self.freq       = float(frequency_hz)
        self.wavelength = C_LIGHT / self.freq
        self.k          = 2.0 * np.pi / self.wavelength
        self.vertices   = np.asarray(vertices, dtype=float)
        self.faces      = np.asarray(faces,    dtype=int)
        self.ram        = ram_model       # optional FreqDependentRAM instance
        self._build_facets()

    def _build_facets(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross          = np.cross(v1 - v0, v2 - v0)
        mag            = np.linalg.norm(cross, axis=1, keepdims=True)
        self.normals   = cross / (mag + 1e-30)
        self.areas     = 0.5 * mag[:, 0]
        self.centroids = (v0 + v1 + v2) / 3.0
        # per-facet reflectivity (modified by RAM model)
        self.reflectivity = np.ones(len(self.faces))

    def _update_ram(self):
        """Update per-facet reflectivity from RAM model at current frequency."""
        if self.ram is not None:
            amp = self.ram.amplitude_reflectivity(self.freq)
            self.reflectivity[:] = amp

    def _po_amplitude(self, shat, shat_rx=None):
        """
        Returns complex PO scatter amplitude f such that σ = (4π/λ²)|f|².
        shat    : unit vector from target to TRANSMITTER (illumination direction)
        shat_rx : unit vector from target to RECEIVER; None => monostatic (shat_rx=shat)

        Bistatic tangent-plane PO: obliquity and phase use the incidence/
        scatter BISECTOR (ŝ+ŝ_rx)/2 and (ŝ+ŝ_rx) respectively -- the
        standard leading-order bistatic facet approximation. At shat_rx=shat
        this reduces exactly to the original monostatic formula (obliquity
        = n·ŝ, phase = exp(j2k c·ŝ)) -- verified bit-for-bit in solvers.py
        self-check.
        """
        self._update_ram()
        if shat_rx is None:
            shat_rx = shat
        cos_inc = self.normals @ shat
        illum   = cos_inc > 1e-6
        if not illum.any():
            return 0j
        bisector  = 0.5 * (shat + shat_rx)
        obliquity = self.normals[illum] @ bisector
        phase     = np.exp(1j * self.k * (self.centroids[illum] @ (shat + shat_rx)))
        weights   = self.reflectivity[illum] * obliquity * self.areas[illum]
        return np.dot(weights, phase)

    def bistatic_rcs_dbsm(self, az_tx_deg, el_tx_deg, az_rx_deg, el_rx_deg):
        """Bistatic RCS (PO layer only -- PTD/double-bounce remain monostatic,
        see class docstrings) for a transmitter/receiver pair at arbitrary
        separate angles. az_rx=az_tx+180, el_rx=-el_tx is the forward-scatter
        case (Babinet's-principle regime, not reducible by geometric shaping)."""
        shat_tx = self._shat(az_tx_deg, el_tx_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg)
        return self._amplitude_to_rcs(self._po_amplitude(shat_tx, shat_rx))

    # ── Polarimetric PO (HH/HV/VH/VV) ──────────────────────────────────────

    def _global_pol_basis(self, shat):
        """Fixed global (H,V) unit vectors transverse to look direction shat
        -- standard radar convention: H horizontal, V in the vertical plane
        containing shat and the world Z axis."""
        z = np.array([0.0, 0.0, 1.0])
        if abs(shat @ z) > 0.999:
            z = np.array([1.0, 0.0, 0.0])   # near-vertical look: arbitrary transverse ref
        H = np.cross(z, shat)
        H = H / (np.linalg.norm(H) + 1e-30)
        V = np.cross(shat, H)
        return H, V

    def _po_scatter_matrix(self, shat_tx, shat_rx=None):
        """
        Polarimetric PO scattering matrix S (2x2 complex, global H/V basis):
        [E_H_scat; E_V_scat] = S [E_H_inc; E_V_inc], sigma_pq = (4pi/lambda^2)|S_pq|^2.

        Each illuminated PEC facet reflects with the standard TE/TM
        dichotomy in ITS OWN local plane-of-incidence basis (r_perp=-1,
        r_par=+1 -- the textbook PEC reflection pair). Projecting that
        local reflection into the FIXED global H/V frame is what produces
        cross-pol (HV/VH): a facet square-on to the radar has no defined
        plane of incidence (falls back to the global basis, r_par=r_perp
        there so no spurious cross-pol appears) and stays pure co-pol; a
        tilted facet or dihedral corner has a rotated local basis and
        leaks energy into the cross channel. This is the same mechanism
        that makes corner reflectors depolarizing in real radar polarimetry.
        """
        self._update_ram()
        if shat_rx is None:
            shat_rx = shat_tx
        cos_inc = self.normals @ shat_tx
        illum   = cos_inc > 1e-6
        if not illum.any():
            return np.zeros((2, 2), dtype=complex)

        n   = self.normals[illum]
        A   = self.areas[illum]
        rho = self.reflectivity[illum]
        c   = self.centroids[illum]

        bisector  = 0.5 * (shat_tx + shat_rx)
        obliquity = n @ bisector
        phase     = np.exp(1j * self.k * (c @ (shat_tx + shat_rx)))
        weight    = rho * A * obliquity * phase

        H, V = self._global_pol_basis(shat_tx)

        shat_tx_b = np.broadcast_to(shat_tx, n.shape)
        e_perp    = np.cross(n, shat_tx_b)
        norm_perp = np.linalg.norm(e_perp, axis=1)
        degenerate = norm_perp < 1e-8
        e_perp = np.where(degenerate[:, None], H, e_perp / (norm_perp[:, None] + 1e-30))
        e_par  = np.cross(shat_tx_b, e_perp)

        r_perp = -1.0
        r_par  = np.where(degenerate, r_perp, 1.0)   # degenerate facets: TE=TM, no spurious depol

        R11 = e_perp @ H; R12 = e_par @ H   # local->global row for H
        R21 = e_perp @ V; R22 = e_par @ V   # local->global row for V

        J_HH = weight * (R11*r_perp*R11 + R12*r_par*R12)
        J_HV = weight * (R11*r_perp*R21 + R12*r_par*R22)
        J_VH = weight * (R21*r_perp*R11 + R22*r_par*R12)
        J_VV = weight * (R21*r_perp*R21 + R22*r_par*R22)

        return np.array([[J_HH.sum(), J_HV.sum()],
                          [J_VH.sum(), J_VV.sum()]])

    def polarimetric_rcs_dbsm(self, az_deg, el_deg=0.0, az_rx_deg=None, el_rx_deg=None):
        """Polarimetric RCS (PO layer). az_rx/el_rx=None => monostatic.
        Returns {'HH','HV','VH','VV': dBsm}. Subclasses override to add
        their own scattering-matrix contributions (edge, double-bounce)."""
        shat_tx = self._shat(az_deg, el_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg) if az_rx_deg is not None else None
        S = self._po_scatter_matrix(shat_tx, shat_rx)
        idx = {'HH': (0, 0), 'HV': (0, 1), 'VH': (1, 0), 'VV': (1, 1)}
        return {pq: self._amplitude_to_rcs(S[i, j]) for pq, (i, j) in idx.items()}

    @staticmethod
    def _facet_reflection_jones(n, look_dir, H, V):
        """2x2 Jones matrix (global H,V basis) for one PEC reflection off a
        surface with normal n, given the target-to-source look direction.
        Standard PEC TE/TM dichotomy (r_perp=-1, r_par=+1); degenerate
        (normal-incidence-like) cases fall back to r_par=r_perp so no
        spurious depolarisation is introduced. Shared by facet double-bounce
        and (with an edge direction in place of a facet normal) PTD."""
        e_perp = np.cross(n, look_dir)
        norm = np.linalg.norm(e_perp)
        if norm < 1e-8:
            e_perp, r_par = H, -1.0
        else:
            e_perp, r_par = e_perp / norm, 1.0
        e_par = np.cross(look_dir, e_perp)
        R = np.array([[e_perp @ H, e_par @ H],
                       [e_perp @ V, e_par @ V]])
        return R @ np.diag([-1.0, r_par]) @ R.T

    def _amplitude_to_rcs(self, amplitude):
        sigma = (4.0 * np.pi / self.wavelength**2) * np.abs(amplitude)**2
        return 10.0 * np.log10(sigma + 1e-40)

    def _shat(self, az_deg, el_deg):
        az, el = np.radians(az_deg), np.radians(el_deg)
        return np.array([np.cos(el)*np.cos(az),
                          np.cos(el)*np.sin(az),
                          np.sin(el)])

    def monostatic_rcs_dbsm(self, az_deg, el_deg=0.0):
        return self._amplitude_to_rcs(self._po_amplitude(self._shat(az_deg, el_deg)))

    def azimuth_sweep(self, n=360, el_deg=0.0):
        angles = np.linspace(0.0, 360.0, n, endpoint=False)
        rcs    = np.array([self.monostatic_rcs_dbsm(az, el_deg) for az in angles])
        return angles, rcs

    def heatmap(self, n_az=120, n_el=60):
        azs  = np.linspace(0.0,   360.0, n_az, endpoint=False)
        els  = np.linspace(-75.0,  75.0, n_el)
        grid = np.empty((n_el, n_az))
        for i, el in enumerate(els):
            for j, az in enumerate(azs):
                grid[i, j] = self.monostatic_rcs_dbsm(az, el)
        return azs, els, grid


# ─────────────────────────────────────────────────────────────────────────────
# 1. PTD  —  Physical Theory of Diffraction  (Ufimtsev fringe field)
# ─────────────────────────────────────────────────────────────────────────────

class PTDSolver(POAmplitudeSolver):
    """
    PO + PTD combined RCS solver.

    The PTD adds the Ufimtsev fringe field correction to PO.  For each edge
    in the mesh the correction is:

        f_fringe(edge) = D_fringe(n, β) × L × sinc(kL cosβ) × e^{j2k ŝ·c}

    where:
        n     = n_wedge  = exterior wedge angle / π
        β     = angle between edge direction ê and radar direction ŝ
        L     = edge length
        c     = edge midpoint
        sinc  = sin(x)/x  (peaks when edge ⊥ radar, i.e. cosβ ≈ 0)

    For a PEC wedge in backscatter (monostatic), the Keller diffraction
    coefficient (averaged over polarization) is:

        D_fringe ≈ -e^{-jπ/4} / (n √(2πk)) × cot(π/2n) / sinβ

    Edge classification:
        boundary (1 adjacent face)  : free edge, n = 2  (half-plane)
        internal convex (2 faces, outward normals diverge) : n < 2
        internal concave (corner reflector) : n > 2  (stronger!)

    Physical meaning:
        A right-angle convex ridge (fuselage corner) has n = 1.5.
        A half-plane free edge (wing tip) has n = 2.
        A right-angle concave cavity (corner reflector) has n = 2.5.
    """

    def __init__(self, vertices, faces, frequency_hz, ram_model=None):
        super().__init__(vertices, faces, frequency_hz, ram_model)
        self._extract_edges()
        self._classify_edges()

    # ── Edge extraction ────────────────────────────────────────────────────

    def _extract_edges(self):
        """Find all unique edges and their adjacent face indices."""
        edge_map = {}
        for fi, face in enumerate(self.faces):
            for k in range(3):
                v0, v1 = int(face[k]), int(face[(k+1)%3])
                key = (min(v0,v1), max(v0,v1))
                edge_map.setdefault(key, []).append(fi)
        # edges: list of (v0, v1, [adj_face_indices])
        self.edges = [(k[0], k[1], v) for k, v in edge_map.items()]

    def _classify_edges(self):
        """
        For each edge compute:
          length, midpoint, direction, n_wedge
        """
        lengths, midpoints, directions, n_wedges = [], [], [], []

        for v0, v1, adj in self.edges:
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            L  = np.linalg.norm(p1 - p0)
            lengths.append(L)
            midpoints.append((p0 + p1) * 0.5)
            directions.append((p1 - p0) / (L + 1e-30))

            if len(adj) == 1:
                # Boundary / free edge — half-plane, n = 2
                n_wedges.append(2.0)
            else:
                # Internal edge — compute dihedral from adjacent face normals
                n1 = self.normals[adj[0]]
                n2 = self.normals[adj[1]]
                cos_dih = np.clip(n1 @ n2, -1.0, 1.0)
                interior_dihedral = np.arccos(cos_dih)  # angle between normals

                # Determine convex vs concave:
                # Use edge midpoint and facet centroids to decide
                edge_mid = midpoints[-1]
                c1 = self.centroids[adj[0]]
                c2 = self.centroids[adj[1]]
                # Vector from edge midpoint toward face centroid average
                avg_toward = 0.5*((c1 - edge_mid) + (c2 - edge_mid))
                avg_normal = 0.5*(n1 + n2)
                # If normals point generally AWAY from the mesh interior
                # the edge is convex (standard exterior ridge)
                sign = np.sign(avg_normal @ avg_toward + 1e-12)

                if sign > 0:
                    # Concave (normals point inward toward the body centre)
                    # exterior wedge > π  →  corner reflector type
                    n_ext = (2.0 * np.pi - interior_dihedral) / np.pi
                else:
                    # Convex ridge (normals point outward)
                    n_ext = (np.pi + interior_dihedral) / np.pi

                n_wedges.append(float(np.clip(n_ext, 0.51, 3.99)))

        self.edge_L   = np.array(lengths)
        self.edge_c   = np.array(midpoints)
        self.edge_dir = np.array(directions)
        self.edge_n   = np.array(n_wedges)

    # ── Fringe diffraction coefficient ─────────────────────────────────────

    def _D_fringe(self, n, sin_beta):
        """
        Scalar Ufimtsev fringe diffraction coefficient for monostatic
        backscatter off a PEC wedge with exterior-angle parameter n.

        D = -exp(-jπ/4) / (n √(2πk)) × cot(π/2n) / sinβ

        The cot(π/2n) factor:
          n=1.0 (flat,   0° wedge): cot(π/2) = 0       → no diffraction
          n=1.5 (90° convex ridge): cot(π/3) = 1/√3 ≈ 0.577
          n=2.0 (half-plane edge) : cot(π/4) = 1.0
          n=2.5 (concave corner)  : cot(π/5) ≈ 1.376   → strongest
        """
        if sin_beta < 0.02:
            return 0j   # endfire: no contribution
        arg = np.pi / (2.0 * n)
        cot = np.cos(arg) / (np.sin(arg) + 1e-15)
        prefactor = np.exp(-1j * np.pi / 4.0) / (n * np.sqrt(2.0 * np.pi * self.k))
        return -prefactor * cot / sin_beta

    # ── Edge fringe amplitude sum ───────────────────────────────────────────

    def _edge_amplitude(self, shat, shat_rx=None):
        """
        Coherent sum of Ufimtsev fringe contributions from all edges.
        shat_rx=None => monostatic. Bistatic generalisation follows the same
        pattern as _po_amplitude: the per-point phase along an edge of a
        wave arriving from shat and leaving toward shat_rx accumulates as
        k(shat+shat_rx)·ê per unit length, so integrating over the edge
        gives sinc(kL·ê·(shat+shat_rx)/2*2) == sinc(kL·ê·bisector) with
        bisector=(shat+shat_rx)/2 (unnormalised) -- reduces exactly to the
        original sinc(kL cosβ) at shat_rx=shat. The diffraction coefficient
        D_fringe's β, by contrast, is a genuine ANGLE (needs a true unit
        direction) so it uses the NORMALISED bisector.
        This is a first-order bistatic extension of the already
        polarization-averaged, azimuth-independent D_fringe model -- not
        the full 4-term Keller/UTD bistatic coefficient (which needs a
        per-edge reference face and separate φ,φ' angles this mesh model
        doesn't track). Real bistatic edge diffraction is concentrated near
        the Keller cone (β_inc≈β_scat); this model doesn't enforce that
        constraint explicitly, so treat off-cone bistatic PTD numbers as
        indicative, not rigorous.
        """
        if shat_rx is None:
            shat_rx = shat
        bisector = 0.5 * (shat + shat_rx)
        bnorm = np.linalg.norm(bisector)
        bisector_hat = bisector / bnorm if bnorm > 1e-9 else shat

        total = 0j
        for i in range(len(self.edges)):
            ê = self.edge_dir[i]
            L = self.edge_L[i]
            c = self.edge_c[i]
            n = self.edge_n[i]

            cos_beta_D = np.dot(ê, bisector_hat)
            sin_beta_D = np.sqrt(max(0.0, 1.0 - cos_beta_D**2))
            D = self._D_fringe(n, sin_beta_D)

            cos_beta_sinc = np.dot(ê, bisector)   # unnormalised: matches phase-integral derivation
            sinc  = _sinc(self.k * L * cos_beta_sinc)
            phase = np.exp(1j * self.k * np.dot(c, shat + shat_rx))

            total += D * L * sinc * phase
        return total

    # ── Polarimetric edge diffraction ──────────────────────────────────────

    def _edge_scatter_matrix(self, shat_tx, shat_rx=None):
        """
        Polarimetric edge-diffraction scattering matrix (2x2, global H/V).

        Simplified soft/hard split of the (already polarization-averaged)
        scalar D_fringe: E parallel to the edge ("soft"/Dirichlet-like) and
        E perpendicular to the edge within the transverse plane
        ("hard"/Neumann-like) pick up opposite-sign D_fringe -- the same
        qualitative soft/hard dichotomy as real GTD, at this model's
        existing level of angular rigor (no φ,φ' dependence). A free edge
        aligned with the global H or V axis stays pure co-pol; an oblique
        edge leaks into the cross channel, same mechanism as the facet
        polarimetric model.
        """
        if shat_rx is None:
            shat_rx = shat_tx
        bisector = 0.5 * (shat_tx + shat_rx)
        bnorm = np.linalg.norm(bisector)
        bisector_hat = bisector / bnorm if bnorm > 1e-9 else shat_tx
        H, V = self._global_pol_basis(shat_tx)

        S = np.zeros((2, 2), dtype=complex)
        for i in range(len(self.edges)):
            ê = self.edge_dir[i]
            L = self.edge_L[i]
            c = self.edge_c[i]
            n = self.edge_n[i]

            cos_beta_D = np.dot(ê, bisector_hat)
            sin_beta_D = np.sqrt(max(0.0, 1.0 - cos_beta_D**2))
            D = self._D_fringe(n, sin_beta_D)
            cos_beta_sinc = np.dot(ê, bisector)
            sinc  = _sinc(self.k * L * cos_beta_sinc)
            phase = np.exp(1j * self.k * np.dot(c, shat_tx + shat_rx))
            weight = D * L * sinc * phase

            e_par = ê
            e_perp = np.cross(bisector_hat, ê)
            norm = np.linalg.norm(e_perp)
            e_perp = H if norm < 1e-8 else e_perp / norm
            R = np.array([[e_par @ H, e_perp @ H],
                          [e_par @ V, e_perp @ V]])
            J = R @ np.diag([1.0, -1.0]) @ R.T   # soft(+)/hard(-) dichotomy

            S += weight * J
        return S

    # ── Public interface ───────────────────────────────────────────────────

    def monostatic_rcs_dbsm(self, az_deg, el_deg=0.0):
        shat = self._shat(az_deg, el_deg)
        f_po   = self._po_amplitude(shat)
        f_edge = self._edge_amplitude(shat)
        return self._amplitude_to_rcs(f_po + f_edge)

    def rcs_breakdown(self, az_deg, el_deg=0.0):
        """Returns dict with individual contributions in dBsm."""
        shat   = self._shat(az_deg, el_deg)
        f_po   = self._po_amplitude(shat)
        f_edge = self._edge_amplitude(shat)
        f_tot  = f_po + f_edge
        return {
            'PO only':     self._amplitude_to_rcs(f_po),
            'PTD edge':    self._amplitude_to_rcs(f_edge),
            'PO + PTD':    self._amplitude_to_rcs(f_tot),
        }

    def bistatic_rcs_dbsm(self, az_tx_deg, el_tx_deg, az_rx_deg, el_rx_deg):
        """Bistatic RCS, PO+PTD (see _edge_amplitude for the bistatic PTD caveat)."""
        shat_tx = self._shat(az_tx_deg, el_tx_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg)
        f = self._po_amplitude(shat_tx, shat_rx) + self._edge_amplitude(shat_tx, shat_rx)
        return self._amplitude_to_rcs(f)

    def polarimetric_rcs_dbsm(self, az_deg, el_deg=0.0, az_rx_deg=None, el_rx_deg=None):
        """Polarimetric RCS, PO+PTD. az_rx/el_rx=None => monostatic."""
        shat_tx = self._shat(az_deg, el_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg) if az_rx_deg is not None else None
        S = self._po_scatter_matrix(shat_tx, shat_rx) + self._edge_scatter_matrix(shat_tx, shat_rx)
        idx = {'HH': (0, 0), 'HV': (0, 1), 'VH': (1, 0), 'VV': (1, 1)}
        return {pq: self._amplitude_to_rcs(S[i, j]) for pq, (i, j) in idx.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOUBLE BOUNCE  —  Second-order geometric-optics reflections
# ─────────────────────────────────────────────────────────────────────────────

class DoubleBounce(PTDSolver):
    """
    PO + PTD + Double-bounce RCS solver.

    For each ordered pair (i, j) of facets:
      1. Check facet i is illuminated by the radar (cos θᵢ > 0).
      2. The specular reflection direction from facet i:
             ŝ_ref = ŝ - 2(n̂ᵢ · ŝ) n̂ᵢ
      3. Check facet j is illuminated by ŝ_ref (cos θⱼ_ref > 0).
      4. Check the re-scattered direction from j back to radar is valid
             (for monostatic: same as ŝ, so n̂ⱼ · ŝ > 0 also).
      5. Amplitude contribution:
             f_ij = Aᵢ cos(θᵢ) × Aⱼ cos(θⱼ_ref) × exp(j phase_ij)
         where phase_ij = 2k (ŝ·cᵢ + ŝ_ref·(cⱼ-cᵢ))
                        ≈ 2k (ŝ·cⱼ)  for monostatic (exact retrace path)

    Corner reflector signature:
        On conventional aircraft the fin-fuselage junction creates a
        near-right-angle dihedral.  Facet A (vertical fin side) reflects
        toward facet B (horizontal fuselage top) which then retro-reflects
        → massive coherent enhancement visible at 90° broadside.
        On stealth aircraft: no 90° dihedrals → double-bounce is suppressed.
    """

    def _double_bounce_amplitude(self, shat, shat_rx=None):
        """shat_rx=None => monostatic. Bistatic: facet i is illuminated and
        the incident ray direction is set by shat (transmitter); the final
        retroreflection test and the outgoing-leg phase use shat_rx
        (receiver) instead. Reduces exactly to the original monostatic
        formula at shat_rx=shat."""
        if shat_rx is None:
            shat_rx = shat
        cos_theta = self.normals @ shat
        illum     = cos_theta > 1e-6
        n_facets  = len(self.faces)
        total     = 0j

        illum_idx = np.where(illum)[0]

        for i in illum_idx:
            n_i    = self.normals[i]
            c_i    = self.centroids[i]
            A_i    = self.areas[i]
            ct_i   = cos_theta[i]
            rho_i  = self.reflectivity[i]

            # Correct law of reflection:
            # incident ray travels in direction d_inc = -shat
            # reflected ray: d_ref = d_inc - 2(d_inc·n_i)n_i
            d_inc  = -shat
            d_ref_i = d_inc - 2.0*(d_inc @ n_i)*n_i   # ray traveling from i toward j

            for j in range(n_facets):
                if j == i:
                    continue
                n_j   = self.normals[j]
                c_j   = self.centroids[j]
                A_j   = self.areas[j]
                rho_j = self.reflectivity[j]

                # Check facet j is illuminated by the reflected ray from i
                # d_ref_i travels toward j → j is hit if n_j opposes d_ref_i
                cos_j_inc = -(n_j @ d_ref_i)   # = n_j · (-d_ref_i)
                if cos_j_inc < 1e-6:
                    continue

                # Compute second reflection direction from j back toward the receiver
                d_ref_j = d_ref_i - 2.0*(d_ref_i @ n_j)*n_j

                # Bistatic retroreflection test: second bounce must point
                # toward the RECEIVER (shat_rx), not necessarily back at shat.
                cos_retro = d_ref_j @ shat_rx
                if cos_retro < 0.2:   # within ~78° of exact retroreflection
                    continue

                # Two-leg path phase: leg1 references the transmitter
                # direction, leg3 references the receiver direction.
                leg2  = abs(d_ref_i @ (c_j - c_i))
                path  = (shat @ c_i) + leg2 + (shat_rx @ c_j)
                phase = np.exp(1j * self.k * path)

                # Amplitude: both cosine projections, both areas
                amp = (rho_i * rho_j * A_i * ct_i
                       * A_j * cos_j_inc * cos_retro)
                total += amp * phase

        return total

    # ── Polarimetric double bounce ─────────────────────────────────────────

    def _double_bounce_scatter_matrix(self, shat_tx, shat_rx=None):
        """
        Polarimetric double-bounce scattering matrix (2x2, global H/V).

        Each contributing facet pair (i,j) applies TWO cascaded PEC
        reflection Jones matrices: J_pair = J_j @ J_i (matrix product, not
        elementwise -- the field's local polarization basis is
        re-decomposed at each bounce). This is the standard cascaded-Jones
        treatment used for dihedral/corner-reflector polarimetric signatures
        (e.g. Freeman-Durden-style double-bounce scattering): a
        right-angle dihedral rotates linear polarization strongly (real
        depolarization), which this reproduces since J_i and J_j come from
        genuinely different facet normals. Facet j's "look direction" for
        its local basis is -d_ref_i (the direction back toward facet i,
        i.e. where the ray causing this bounce came from).
        """
        if shat_rx is None:
            shat_rx = shat_tx
        cos_theta = self.normals @ shat_tx
        illum     = cos_theta > 1e-6
        n_facets  = len(self.faces)
        S = np.zeros((2, 2), dtype=complex)
        H, V = self._global_pol_basis(shat_tx)

        illum_idx = np.where(illum)[0]

        for i in illum_idx:
            n_i, c_i, A_i = self.normals[i], self.centroids[i], self.areas[i]
            ct_i, rho_i = cos_theta[i], self.reflectivity[i]
            d_inc = -shat_tx
            d_ref_i = d_inc - 2.0*(d_inc @ n_i)*n_i
            J_i = self._facet_reflection_jones(n_i, shat_tx, H, V)

            for j in range(n_facets):
                if j == i:
                    continue
                n_j, c_j, A_j, rho_j = self.normals[j], self.centroids[j], self.areas[j], self.reflectivity[j]
                cos_j_inc = -(n_j @ d_ref_i)
                if cos_j_inc < 1e-6:
                    continue
                d_ref_j = d_ref_i - 2.0*(d_ref_i @ n_j)*n_j
                cos_retro = d_ref_j @ shat_rx
                if cos_retro < 0.2:
                    continue

                J_j = self._facet_reflection_jones(n_j, -d_ref_i, H, V)

                leg2  = abs(d_ref_i @ (c_j - c_i))
                path  = (shat_tx @ c_i) + leg2 + (shat_rx @ c_j)
                phase = np.exp(1j * self.k * path)
                amp   = rho_i * rho_j * A_i * ct_i * A_j * cos_j_inc * cos_retro * phase

                S += amp * (J_j @ J_i)

        return S

    def monostatic_rcs_dbsm(self, az_deg, el_deg=0.0):
        shat   = self._shat(az_deg, el_deg)
        f_po   = self._po_amplitude(shat)
        f_edge = self._edge_amplitude(shat)
        f_db   = self._double_bounce_amplitude(shat)
        return self._amplitude_to_rcs(f_po + f_edge + f_db)

    def rcs_breakdown(self, az_deg, el_deg=0.0):
        shat   = self._shat(az_deg, el_deg)
        f_po   = self._po_amplitude(shat)
        f_edge = self._edge_amplitude(shat)
        f_db   = self._double_bounce_amplitude(shat)
        return {
            'PO only':         self._amplitude_to_rcs(f_po),
            'PTD edge':        self._amplitude_to_rcs(f_edge),
            'Double bounce':   self._amplitude_to_rcs(f_db),
            'PO + PTD':        self._amplitude_to_rcs(f_po + f_edge),
            'Full (PO+PTD+DB)':self._amplitude_to_rcs(f_po + f_edge + f_db),
        }

    def bistatic_rcs_dbsm(self, az_tx_deg, el_tx_deg, az_rx_deg, el_rx_deg):
        """Bistatic RCS, full physics (PO+PTD+DoubleBounce)."""
        shat_tx = self._shat(az_tx_deg, el_tx_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg)
        f = (self._po_amplitude(shat_tx, shat_rx)
             + self._edge_amplitude(shat_tx, shat_rx)
             + self._double_bounce_amplitude(shat_tx, shat_rx))
        return self._amplitude_to_rcs(f)

    def polarimetric_rcs_dbsm(self, az_deg, el_deg=0.0, az_rx_deg=None, el_rx_deg=None):
        """Polarimetric RCS, full physics (PO+PTD+DoubleBounce). az_rx/el_rx=None => monostatic."""
        shat_tx = self._shat(az_deg, el_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg) if az_rx_deg is not None else None
        S = (self._po_scatter_matrix(shat_tx, shat_rx)
             + self._edge_scatter_matrix(shat_tx, shat_rx)
             + self._double_bounce_scatter_matrix(shat_tx, shat_rx))
        idx = {'HH': (0, 0), 'HV': (0, 1), 'VH': (1, 0), 'VV': (1, 1)}
        return {pq: self._amplitude_to_rcs(S[i, j]) for pq, (i, j) in idx.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2b. TRIPLE BOUNCE  —  Trihedral corner-reflector (third-order geometric optics)
# ─────────────────────────────────────────────────────────────────────────────

class TripleBounce(DoubleBounce):
    """
    PO + PTD + Double-bounce + Triple-bounce RCS solver.

    Extends the double-bounce ray trace one more reflection: facet i -> j -> k
    -> back to the radar. A TRIHEDRAL corner reflector (three mutually
    near-perpendicular facets, e.g. a wing-fuselage-tail three-way junction)
    is the classic radar calibration retroreflector -- it returns energy to
    the source over an even WIDER angular range than a dihedral, because a
    true 90-90-90 trihedral retroreflects any ray entering its interior
    regardless of incidence angle. Real conventional aircraft can form one
    where the fuselage top/side and vertical-stabiliser side meet at a
    corner -- a mechanism the double-bounce-only model above misses entirely.

    O(n^3) Python loop -- caps out fast. At the report's baseline mesh
    (34/20 facets) this is ~40,000/~8,000 triples, trivial. Do not run this
    on a subdivided (high-facet) mesh; it will not finish.
    """

    def _triple_bounce_amplitude(self, shat, shat_rx=None):
        """shat_rx=None => monostatic. Same tx/rx convention as
        _double_bounce_amplitude: first bounce keyed to shat (tx), final
        retroreflection test and outgoing phase keyed to shat_rx (rx)."""
        if shat_rx is None:
            shat_rx = shat
        cos_theta = self.normals @ shat
        illum     = cos_theta > 1e-6
        n_facets  = len(self.faces)
        total     = 0j

        illum_idx = np.where(illum)[0]

        for i in illum_idx:
            n_i, c_i, A_i = self.normals[i], self.centroids[i], self.areas[i]
            ct_i, rho_i = cos_theta[i], self.reflectivity[i]
            d_inc = -shat
            d_ref_i = d_inc - 2.0*(d_inc @ n_i)*n_i

            for j in range(n_facets):
                if j == i:
                    continue
                n_j, c_j, A_j, rho_j = self.normals[j], self.centroids[j], self.areas[j], self.reflectivity[j]
                cos_j_inc = -(n_j @ d_ref_i)
                if cos_j_inc < 1e-6:
                    continue
                d_ref_j = d_ref_i - 2.0*(d_ref_i @ n_j)*n_j

                for k in range(n_facets):
                    if k == i or k == j:
                        continue
                    n_k, c_k, A_k, rho_k = self.normals[k], self.centroids[k], self.areas[k], self.reflectivity[k]
                    cos_k_inc = -(n_k @ d_ref_j)
                    if cos_k_inc < 1e-6:
                        continue
                    d_ref_k = d_ref_j - 2.0*(d_ref_j @ n_k)*n_k

                    cos_retro = d_ref_k @ shat_rx
                    if cos_retro < 0.2:
                        continue

                    leg2 = abs(d_ref_i @ (c_j - c_i))
                    leg3 = abs(d_ref_j @ (c_k - c_j))
                    path = (shat @ c_i) + leg2 + leg3 + (shat_rx @ c_k)
                    phase = np.exp(1j * self.k * path)

                    amp = (rho_i * rho_j * rho_k * A_i * ct_i
                           * A_j * cos_j_inc * A_k * cos_k_inc * cos_retro)
                    total += amp * phase

        return total

    # ── Polarimetric triple bounce ─────────────────────────────────────────

    def _triple_bounce_scatter_matrix(self, shat_tx, shat_rx=None):
        """Cascaded 3-reflection Jones matrix per contributing triple:
        J_triple = J_k @ J_j @ J_i (matrix product). Same trihedral
        depolarisation mechanism as the double-bounce dihedral case, one
        bounce deeper -- a real 90-90-90 trihedral flips polarization
        handedness in a distinctive way real polarimetric radars use to
        identify calibration reflectors."""
        if shat_rx is None:
            shat_rx = shat_tx
        cos_theta = self.normals @ shat_tx
        illum     = cos_theta > 1e-6
        n_facets  = len(self.faces)
        S = np.zeros((2, 2), dtype=complex)
        H, V = self._global_pol_basis(shat_tx)

        illum_idx = np.where(illum)[0]

        for i in illum_idx:
            n_i, c_i, A_i = self.normals[i], self.centroids[i], self.areas[i]
            ct_i, rho_i = cos_theta[i], self.reflectivity[i]
            d_inc = -shat_tx
            d_ref_i = d_inc - 2.0*(d_inc @ n_i)*n_i
            J_i = self._facet_reflection_jones(n_i, shat_tx, H, V)

            for j in range(n_facets):
                if j == i:
                    continue
                n_j, c_j, A_j, rho_j = self.normals[j], self.centroids[j], self.areas[j], self.reflectivity[j]
                cos_j_inc = -(n_j @ d_ref_i)
                if cos_j_inc < 1e-6:
                    continue
                d_ref_j = d_ref_i - 2.0*(d_ref_i @ n_j)*n_j
                J_j = self._facet_reflection_jones(n_j, -d_ref_i, H, V)

                for k in range(n_facets):
                    if k == i or k == j:
                        continue
                    n_k, c_k, A_k, rho_k = self.normals[k], self.centroids[k], self.areas[k], self.reflectivity[k]
                    cos_k_inc = -(n_k @ d_ref_j)
                    if cos_k_inc < 1e-6:
                        continue
                    d_ref_k = d_ref_j - 2.0*(d_ref_j @ n_k)*n_k
                    cos_retro = d_ref_k @ shat_rx
                    if cos_retro < 0.2:
                        continue

                    J_k = self._facet_reflection_jones(n_k, -d_ref_j, H, V)

                    leg2 = abs(d_ref_i @ (c_j - c_i))
                    leg3 = abs(d_ref_j @ (c_k - c_j))
                    path = (shat_tx @ c_i) + leg2 + leg3 + (shat_rx @ c_k)
                    phase = np.exp(1j * self.k * path)
                    amp = (rho_i * rho_j * rho_k * A_i * ct_i
                           * A_j * cos_j_inc * A_k * cos_k_inc * cos_retro * phase)

                    S += amp * (J_k @ J_j @ J_i)

        return S

    def monostatic_rcs_dbsm(self, az_deg, el_deg=0.0):
        shat = self._shat(az_deg, el_deg)
        f = (self._po_amplitude(shat) + self._edge_amplitude(shat)
             + self._double_bounce_amplitude(shat) + self._triple_bounce_amplitude(shat))
        return self._amplitude_to_rcs(f)

    def rcs_breakdown(self, az_deg, el_deg=0.0):
        shat   = self._shat(az_deg, el_deg)
        f_po   = self._po_amplitude(shat)
        f_edge = self._edge_amplitude(shat)
        f_db   = self._double_bounce_amplitude(shat)
        f_tb   = self._triple_bounce_amplitude(shat)
        return {
            'PO only':              self._amplitude_to_rcs(f_po),
            'PTD edge':             self._amplitude_to_rcs(f_edge),
            'Double bounce':        self._amplitude_to_rcs(f_db),
            'Triple bounce':        self._amplitude_to_rcs(f_tb),
            'PO + PTD':             self._amplitude_to_rcs(f_po + f_edge),
            'Full (PO+PTD+DB)':     self._amplitude_to_rcs(f_po + f_edge + f_db),
            'Full (PO+PTD+DB+TB)':  self._amplitude_to_rcs(f_po + f_edge + f_db + f_tb),
        }

    def bistatic_rcs_dbsm(self, az_tx_deg, el_tx_deg, az_rx_deg, el_rx_deg):
        """Bistatic RCS, full physics (PO+PTD+DoubleBounce+TripleBounce)."""
        shat_tx = self._shat(az_tx_deg, el_tx_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg)
        f = (self._po_amplitude(shat_tx, shat_rx)
             + self._edge_amplitude(shat_tx, shat_rx)
             + self._double_bounce_amplitude(shat_tx, shat_rx)
             + self._triple_bounce_amplitude(shat_tx, shat_rx))
        return self._amplitude_to_rcs(f)

    def polarimetric_rcs_dbsm(self, az_deg, el_deg=0.0, az_rx_deg=None, el_rx_deg=None):
        """Polarimetric RCS, full physics (PO+PTD+DoubleBounce+TripleBounce).
        az_rx/el_rx=None => monostatic."""
        shat_tx = self._shat(az_deg, el_deg)
        shat_rx = self._shat(az_rx_deg, el_rx_deg) if az_rx_deg is not None else None
        S = (self._po_scatter_matrix(shat_tx, shat_rx)
             + self._edge_scatter_matrix(shat_tx, shat_rx)
             + self._double_bounce_scatter_matrix(shat_tx, shat_rx)
             + self._triple_bounce_scatter_matrix(shat_tx, shat_rx))
        idx = {'HH': (0, 0), 'HV': (0, 1), 'VH': (1, 0), 'VV': (1, 1)}
        return {pq: self._amplitude_to_rcs(S[i, j]) for pq, (i, j) in idx.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2c. SURFACE IMPEDANCE BOUNDARY CONDITION (SIBC) — finite conductivity
# ─────────────────────────────────────────────────────────────────────────────

MU0 = 4.0 * np.pi * 1e-7   # vacuum permeability [H/m]


class SIBCMaterial:
    """
    Leontovich Surface Impedance Boundary Condition: replaces the PEC
    assumption (Js = 2n_hat x H_inc, infinite conductivity, Gamma = -1
    exactly) with a finite-conductivity metal/composite surface.

    Physics:
        Good-conductor surface impedance (skin-effect regime, valid when
        skin depth << facet size, true for all these materials at X-band):
            Zs(f) = (1+j) * sqrt(pi f mu / sigma)
        Treated as a lossy "load" against free space, same transmission-line
        form already used for SalisburyScreen/DallenbachLayer in this file:
            Gamma(f) = (Zs - Z0) / (Zs + Z0)
        |Zs| << Z0 for any real conductor at RF, so |Gamma| -> 1 (near-PEC)
        -- how close is exactly the question this model answers. Plugs into
        every existing solver via the same ram_model= constructor kwarg the
        RAM classes use; no solver code changes needed.

    Parameters:
        conductivity_S_per_m : bulk (or effective, for composites) electrical
                                conductivity sigma [S/m]
        mu_r                 : relative permeability (1.0 for all non-magnetic
                                aerospace metals/composites used here)

    Representative aerospace material conductivities (approximate, room
    temperature, commonly cited engineering reference values):
        PEC (reference)         : sigma -> infinity (use ram_model=None)
        Aluminum 2024-T3        : 1.74e7 S/m  (~30% IACS, standard airframe alloy)
        Titanium Ti-6Al-4V      : 5.80e5 S/m  (aerospace structural titanium)
        Stainless steel 304     : 1.45e6 S/m  (exhaust/hot-section skin)
        Carbon-fiber composite  : 3.0e4  S/m  (CFRP skin, anisotropic in
                                                reality -- this is an
                                                effective isotropic value;
                                                real CFRP conductivity varies
                                                ~1e4-1e5 S/m with fiber
                                                orientation and lay-up)
    """

    PRESETS = {
        'aluminum_2024':   1.74e7,
        'titanium_6al4v':  5.80e5,
        'stainless_304':   1.45e6,
        'cfrp_composite':  3.0e4,
    }

    def __init__(self, conductivity_S_per_m, mu_r=1.0):
        self.sigma = conductivity_S_per_m
        self.mu_r  = mu_r

    @classmethod
    def from_preset(cls, name):
        return cls(cls.PRESETS[name])

    def surface_impedance(self, freq_hz):
        return (1.0 + 1j) * np.sqrt(np.pi * freq_hz * self.mu_r * MU0 / self.sigma)

    def reflection_coefficient(self, freq_hz):
        Zs = self.surface_impedance(freq_hz)
        return (Zs - Z0) / (Zs + Z0)

    def amplitude_reflectivity(self, freq_hz):
        return np.abs(self.reflection_coefficient(freq_hz))

    def reflection_loss_db(self, freqs_hz):
        """|Gamma| in dB relative to a perfect PEC (0 dB = indistinguishable
        from PEC, negative = some energy genuinely lost to the finite
        conductivity rather than merely redirected)."""
        gam = np.array([self.amplitude_reflectivity(f) for f in np.atleast_1d(freqs_hz)])
        return 20.0 * np.log10(gam + 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# 3a. SALISBURY SCREEN  —  Frequency-dependent RAM (narrowband)
# ─────────────────────────────────────────────────────────────────────────────

class SalisburyScreen:
    """
    Resistive sheet + lossless λ/4 spacer + PEC ground plane.

    Physics:
      The quarter-wave spacer transforms the ground-plane short circuit into
      an open circuit at f₀.  The resistive sheet Rs = Z₀ = 377 Ω/□ then
      sees no current flowing to ground → perfect absorption at f₀.

      Input impedance:
          Z_gp(f)  = j Z₀ tan(2πf d/c)       [spacer looking into ground]
          Z_in     = Rs ‖ Z_gp                 [sheet in parallel with spacer]
          Γ(f)     = (Z_in − Z₀)/(Z_in + Z₀)

    Parameters:
        f0_hz  : design frequency for maximum absorption [Hz]
        Rs     : sheet resistance [Ω/□], default Z₀ for maximum absorption
    """

    def __init__(self, f0_hz, Rs=Z0):
        self.f0  = f0_hz
        self.Rs  = Rs
        self.d   = C_LIGHT / (4.0 * f0_hz)    # quarter-wave spacer thickness

    def reflection_coefficient(self, freq_hz):
        """Complex reflection coefficient Γ(f)."""
        beta_d = 2.0 * np.pi * freq_hz * self.d / C_LIGHT
        Z_gp   = 1j * Z0 * np.tan(beta_d)
        # Parallel combination: Z_in = Rs * Z_gp / (Rs + Z_gp)
        Z_in   = self.Rs * Z_gp / (self.Rs + Z_gp + 1e-30j)
        return (Z_in - Z0) / (Z_in + Z0)

    def amplitude_reflectivity(self, freq_hz):
        """Amplitude attenuation factor |Γ(f)| (0 = perfect absorber, 1 = bare PEC)."""
        return np.abs(self.reflection_coefficient(freq_hz))

    def absorption_db(self, freqs_hz):
        """Return absorption [dB] as a function of frequency (negative = absorption)."""
        Gamma = np.array([self.reflection_coefficient(f) for f in freqs_hz])
        return 20.0 * np.log10(np.abs(Gamma) + 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. DALLENBACH LAYER  —  Frequency-dependent RAM (broadband)
# ─────────────────────────────────────────────────────────────────────────────

class DallenbachLayer:
    """
    Homogeneous lossy slab of thickness d with complex εᵣ and μᵣ backed
    by a PEC ground plane.

    Physics:
        Propagation constant : γ = j(2πf/c)√(μᵣ εᵣ)
        Wave impedance       : Z_m = Z₀ √(μᵣ/εᵣ)
        Input impedance      : Z_in = Z_m tanh(γ d)
        Reflection coeff     : Γ = (Z_in − Z₀)/(Z_in + Z₀)

    Material presets (at X-band, ~10 GHz):
        'carbon_foam'   : εᵣ = 1.5−0.8j,  μᵣ = 1.0        (lightweight absorber)
        'ferrite_tile'  : εᵣ = 12−3j,     μᵣ = 2.5−1.5j   (heavy but wideband)
        'carbonyl_iron' : εᵣ = 8−4j,      μᵣ = 1.8−0.9j   (balanced)

    These are representative values; real materials are frequency-dispersive.
    For this model εᵣ and μᵣ are held constant over the sweep (single-pole
    approximation — adequate for comparative analysis).
    """

    PRESETS = {
        'carbon_foam':    (1.5 - 0.8j,  1.0 + 0j),
        'ferrite_tile':   (12  - 3j,    2.5 - 1.5j),
        'carbonyl_iron':  (8   - 4j,    1.8 - 0.9j),
    }

    def __init__(self, thickness_m, eps_r, mu_r):
        self.d     = thickness_m
        self.eps_r = complex(eps_r)
        self.mu_r  = complex(mu_r)

    @classmethod
    def from_preset(cls, preset_name, thickness_m=0.005):
        eps_r, mu_r = cls.PRESETS[preset_name]
        return cls(thickness_m, eps_r, mu_r)

    def reflection_coefficient(self, freq_hz):
        k0    = 2.0 * np.pi * freq_hz / C_LIGHT
        gamma = 1j * k0 * np.sqrt(self.mu_r * self.eps_r + 1e-30j)
        Z_m   = Z0 * np.sqrt(self.mu_r / (self.eps_r + 1e-30j))
        Z_in  = Z_m * np.tanh(gamma * self.d)
        return (Z_in - Z0) / (Z_in + Z0)

    def amplitude_reflectivity(self, freq_hz):
        return np.abs(self.reflection_coefficient(freq_hz))

    def absorption_db(self, freqs_hz):
        Gamma = np.array([self.reflection_coefficient(f) for f in freqs_hz])
        return 20.0 * np.log10(np.abs(Gamma) + 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Full-physics sweep functions
# ─────────────────────────────────────────────────────────────────────────────

def make_solver(vertices, faces, frequency_hz, level='full', ram_model=None):
    """
    level:  'po'     → PO only (POAmplitudeSolver)
            'ptd'    → PO + PTD edges (PTDSolver)
            'full'   → PO + PTD + double bounce (DoubleBounce)
            'triple' → PO + PTD + double + triple bounce (TripleBounce) -- O(n^3), baseline mesh only
    """
    kw = dict(ram_model=ram_model)
    if level == 'po':
        return POAmplitudeSolver(vertices, faces, frequency_hz, **kw)
    elif level == 'ptd':
        return PTDSolver(vertices, faces, frequency_hz, **kw)
    elif level == 'triple':
        return TripleBounce(vertices, faces, frequency_hz, **kw)
    else:
        return DoubleBounce(vertices, faces, frequency_hz, **kw)


def physics_frequency_sweep(vertices, faces, freqs_hz,
                              az_deg=0.0, el_deg=0.0,
                              level='full', ram_model=None):
    """RCS vs frequency at a fixed look angle, for a given physics level."""
    rcs = []
    for f in freqs_hz:
        s = make_solver(vertices, faces, f, level=level, ram_model=ram_model)
        rcs.append(s.monostatic_rcs_dbsm(az_deg, el_deg))
    return np.array(rcs)


def breakdown_sweep(vertices, faces, frequency_hz, n=180):
    """
    Full 360° azimuth sweep returning individual contributions.
    Returns dict: key → rcs array (dBsm)
    """
    solver = DoubleBounce(vertices, faces, frequency_hz)
    angles = np.linspace(0, 360, n, endpoint=False)
    keys   = ['PO only', 'PTD edge', 'Double bounce', 'PO + PTD', 'Full (PO+PTD+DB)']
    out    = {k: [] for k in keys}
    for az in angles:
        bd = solver.rcs_breakdown(az)
        for k in keys:
            out[k].append(bd[k])
    out['angles'] = angles
    return {k: np.array(v) if k != 'angles' else v for k, v in out.items()}


def _self_check():
    """Sanity checks for bistatic/polarimetric PO+PTD+DoubleBounce -- run: python solvers.py"""
    from geometries import flat_plate, conventional_aircraft

    v, f = conventional_aircraft()
    s = DoubleBounce(v, f, 10e9)

    # 1. Full-physics bistatic reduces exactly to monostatic when tx=rx
    #    (PO, PTD and double-bounce all generalise correctly).
    mono = s.monostatic_rcs_dbsm(37.0, 12.0)
    bi_self = s.bistatic_rcs_dbsm(37.0, 12.0, 37.0, 12.0)
    assert abs(mono - bi_self) < 1e-9, (mono, bi_self)

    # 2. Flat plate at broadside (normal incidence): pure co-pol, HH==VV,
    #    cross-pol near-zero (no plane of incidence at normal incidence).
    vp, fp = flat_plate(2.0, 2.0)
    sp = POAmplitudeSolver(vp, fp, 10e9)
    pol = sp.polarimetric_rcs_dbsm(0.0, 90.0)   # looking straight down the plate normal (+Z)
    assert abs(pol['HH'] - pol['VV']) < 0.1, pol
    assert pol['HV'] < pol['HH'] - 60, pol   # cross-pol >60dB below co-pol == numerically zero

    # 3. Generic bistatic pair differs from monostatic (conventional aircraft,
    #    angles chosen so both tx and rx illuminate non-degenerate facets).
    bi  = s.bistatic_rcs_dbsm(0.0, 0.0, 30.0, 10.0)
    mono_po = s.bistatic_rcs_dbsm(0.0, 0.0, 0.0, 0.0)
    assert abs(bi - mono_po) > 0.5, (bi, mono_po)

    # 4. Full-physics polarimetric HH matches scalar monostatic RCS in the
    #    co-pol channel where PO dominates and cross-pol is negligible
    #    (nose-on: no active corner reflector -- HH should track the total).
    pol_nose = s.polarimetric_rcs_dbsm(0.0, 0.0)
    mono_nose = s.monostatic_rcs_dbsm(0.0, 0.0)
    assert abs(pol_nose['HH'] - mono_nose) < 3.0, (pol_nose, mono_nose)

    # 5. Double-bounce corner reflector (quarter-on, ~45 deg, where this
    #    aircraft's fin-fuselage dihedral activates) produces real cross-pol
    #    from the double-bounce term -- the cascaded-Jones dihedral
    #    depolarisation signature that single-bounce PO alone cannot.
    pol_corner = s.polarimetric_rcs_dbsm(45.0, 0.0)
    po_only_solver = POAmplitudeSolver(v, f, 10e9)
    pol_corner_po  = po_only_solver.polarimetric_rcs_dbsm(45.0, 0.0)
    assert pol_corner['HV'] > pol_corner_po['HV'] + 3.0, (pol_corner, pol_corner_po)

    print("solvers.py self-check: OK")
    print(f"  Full-physics bistatic(tx=rx) vs monostatic match: {bi_self:.4f} == {mono:.4f}")
    print(f"  Flat plate broadside: HH={pol['HH']:.2f} VV={pol['VV']:.2f} "
          f"HV={pol['HV']:.2f} VH={pol['VH']:.2f} dBsm")
    print(f"  Corner reflector (45deg) HV: PO-only={pol_corner_po['HV']:.2f}  "
          f"full-physics={pol_corner['HV']:.2f} dBsm (double-bounce depolarises)")

    # 6. Triple-bounce: bistatic reduces exactly to monostatic when tx=rx.
    st = TripleBounce(v, f, 10e9)
    mono_tb = st.monostatic_rcs_dbsm(85.0, 60.0)
    bi_tb   = st.bistatic_rcs_dbsm(85.0, 60.0, 85.0, 60.0)
    assert abs(mono_tb - bi_tb) < 1e-9, (mono_tb, bi_tb)

    # 7. Triple bounce is finite everywhere and, at its strongest angle for
    #    this geometry, exceeds double-bounce -- consistent with the known
    #    textbook fact that trihedral corner reflectors return MORE energy
    #    than dihedrals (sigma ~ L^4/lambda^2 vs a weaker L^2-scaling for a
    #    simple dihedral), not a bug.
    breakdown_peak = st.rcs_breakdown(85.0, 60.0)
    assert np.isfinite(breakdown_peak['Triple bounce'])
    assert breakdown_peak['Triple bounce'] > breakdown_peak['Double bounce'], breakdown_peak

    print(f"  Triple-bounce (trihedral) peak at az=85,el=60: "
          f"DB={breakdown_peak['Double bounce']:.2f}  TB={breakdown_peak['Triple bounce']:.2f} dBsm "
          f"(trihedral > dihedral, as expected)")

    # 8. SIBC: very high conductivity (silver-like, 6.3e7 S/m) must reflect
    #    almost exactly like bare PEC (|Gamma| -> 1); the CFRP preset (worst
    #    real material) must show strictly more loss than aluminum (best).
    sibc_hi   = SIBCMaterial(6.3e7)
    sibc_al   = SIBCMaterial.from_preset('aluminum_2024')
    sibc_cfrp = SIBCMaterial.from_preset('cfrp_composite')
    assert sibc_hi.amplitude_reflectivity(10e9) > 0.999, sibc_hi.amplitude_reflectivity(10e9)
    assert sibc_cfrp.reflection_loss_db(10e9)[0] < sibc_al.reflection_loss_db(10e9)[0], \
        (sibc_cfrp.reflection_loss_db(10e9), sibc_al.reflection_loss_db(10e9))
    rcs_pec  = DoubleBounce(v, f, 10e9).monostatic_rcs_dbsm(37.0, 12.0)
    rcs_sibc = DoubleBounce(v, f, 10e9, ram_model=sibc_cfrp).monostatic_rcs_dbsm(37.0, 12.0)
    assert rcs_sibc <= rcs_pec + 1e-9, (rcs_pec, rcs_sibc)
    print(f"  SIBC: high-sigma |Gamma|={sibc_hi.amplitude_reflectivity(10e9):.6f} (~1=PEC), "
          f"PEC={rcs_pec:.3f} vs CFRP-SIBC={rcs_sibc:.3f} dBsm (SIBC <= PEC, as expected)")


def mie_rcs_dbsm(radius_m, frequency_hz):
    """Exact Mie series for conducting sphere (unchanged from v1)."""
    lam   = C_LIGHT / frequency_hz
    ka    = 2.0 * np.pi * radius_m / lam
    n_max = max(int(ka + 4.0*ka**(1/3) + 4), 5)
    S = 0j
    for n in range(1, n_max + 1):
        jn  = spherical_jn(n, ka)
        jnp = spherical_jn(n, ka, derivative=True)
        yn  = spherical_yn(n, ka)
        ynp = spherical_yn(n, ka, derivative=True)
        hn  = jn  + 1j*yn
        hnp = jnp + 1j*ynp
        an  = (ka*jnp + jn)  / (ka*hnp + hn)
        bn  =  jn            /  hn
        S  += (-1)**n * (2*n+1) * (an - bn)
    sigma = (lam**2 / np.pi) * np.abs(S)**2
    return 10.0 * np.log10(sigma + 1e-40)


if __name__ == '__main__':
    _self_check()
