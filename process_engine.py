"""
process_engine.py  (v5)
=======================
Bilayer resist + correct Dolan bridge geometry (Bilmes et al. 2101.01453).

Dolan bridge structure (Fig. 1a of arxiv:2101.01453):
─────────────────────────────────────────────────────
  Top view:
    Large left electrode ── bridge slab ── Large right electrode
    Both sides of bridge have open "side trenches" (openings in resist)
    These side trenches are perpendicular to the evaporation direction.

  Parameters:
    bridge_len  : length of bridge along the evaporation (x) direction [nm]
                  = the gap between left and right electrode PMMA walls
    bridge_w    : width of bridge perpendicular to evaporation (y) direction [nm]
                  = width of the bridge slab = defines junction width

  Cross-section along x (evaporation direction):
    z
    │  ██████ PMMA ██████   ██████ PMMA ██████
    │  ██████ PMMA ██████   ██████ PMMA ██████
    │  ─ ─ bridge slab ─ ─  (hangs at z=t_mma to t_mma+t_pmma)
    │                           ↑ MMA air gap below bridge
    │  ██████  MMA  ██████   ██████  MMA  ██████
    │  ──────────────────────────────────────────  z=0 wafer
    └──────────────────────────────────────── x

  The bridge slab spans x ∈ [-bridge_len/2, +bridge_len/2]
  MMA undercut widens the opening to ±(bridge_len/2 + undercut) in x
  Side trench opening (y direction) = bridge_w + 2*undercut in MMA

  Shadow projection is along x (φ=0 is the standard evaporation direction):
    sx = t_resist · tan(θ)
    The junction overlap in x = bridge_len - 2·|sx| ... complicated
    → see junction_area.py for correct formula

Manhattan cross (unchanged):
  φ₁ = 0°, φ₂ = 90°, cross-shaped opening.
"""

from dataclasses import dataclass
import copy
import numpy as np


@dataclass
class ProcessParams:
    """All process parameters (nm / degrees)."""

    # ── Bilayer resist ─────────────────────────────────────────
    # Recipe: arxiv:2101.01453 — PMMA A-4 ~250 nm on MMA EL-13 ~900 nm.
    t_pmma:   float = 250.0   # PMMA top layer [nm]  (no undercut)
    t_mma:    float = 900.0   # MMA  bot layer [nm]  (= bridge underside height / vertical gap)
    undercut: float = 150.0   # MMA one-sided undercut [nm]
    resist_round: float = 0.0 # resist opening corner fillet radius [nm] (0 = sharp)
    resist_round_method: str = "analytic"  # "analytic" (fast, exact continuous
                                            # quarter-circle) or "voxel" (legacy
                                            # K-slab box-stack approximation, slow)
    soft_edge: bool = False    # soft-edge (penumbra) deposition from a finite source
    soft_pattern: str = "rotline"  # e-beam raster pattern (sample_beam_cloud key)
    soft_size:   float = 12.0   # source spot size on the target [mm] (recipe ~10–15)
    soft_L:      float = 550.0  # source→sample throw distance [mm] (Plassys ~550)
    soft_rays:   int = 24       # source-cloud ray count K (finer ⇒ smoother taper, slower)
    soft_supersample_xy: int = 1   # lateral (xy) n×n sub-sampling per band cell
                                    # (smoother in-plane footprint edge; 1 = centre
                                    # point only, unchanged)
    soft_supersample_z:  int = 1   # z-direction n sub-sampling per band cell
                                    # (smoother through-thickness taper; 1 = cell-
                                    # centre z only, unchanged)

    # ── Evaporation 1 ─────────────────────────────────────────
    # Uniaxial tilt ±24° (same azimuth φ, opposite polar θ); 30 nm Al each.
    # t_metal1/t_metal2 (and tri_t1..tri_t4 below) are nominal thicknesses
    # measured ALONG THE BEAM DIRECTION (deposition-rate-monitor convention),
    # not a raw vertical thickness — the actual vertical thickness deposited
    # on a horizontal floor is t_metal·cos(tilt angle), thinner at more
    # oblique tilts.
    angle1:   float = -24.0
    phi1:     float =   0.0
    t_metal1: float =  30.0

    # ── Evaporation 2 ─────────────────────────────────────────
    angle2:   float =  24.0
    phi2:     float =   0.0
    t_metal2: float =  30.0

    # ── Dolan bridge geometry ──────────────────────────────────
    # bridge_len: bridge width along the evap (x) axis = shadow-defining
    #             narrow dimension (≈ critical dimension, 250 nm in recipe).
    bridge_len: float = 250.0   # bridge width   [nm]  (along evap x-axis)
    bridge_w:   float = 250.0   # bridge length  [nm]  (junction width, y-axis)
    # Horizontal opening between each bridge edge and the PMMA wall (per side).
    # Sets the trench window width; the tilted beam must clear this to reach
    # under the bridge.  0 → auto (wide enough for the beam to reach the floor).
    bridge_pmma_gap: float = 0.0  # one-sided bridge↔PMMA opening [nm]; 0 = auto

    # ── Manhattan / double-oblique geometry (arxiv:2605.19590) ─
    # Two perpendicular resist line openings (Manhattan crossing).
    # Each electrode line is narrowed by double-oblique shadowing:
    #   w_narrow = w_open − h·sin(δ) / tan(θ)        (Eq. A6)
    # where θ = deposition tilt from substrate normal,
    #       δ = in-plane offset between beam azimuth and line direction,
    #       h = upper (imaging) resist thickness.
    manhattan_wx: float = 600.0   # designed opening width of x-running arm [nm]
    manhattan_wy: float = 600.0   # designed opening width of y-running arm [nm]
    manhattan_theta: float = 60.0 # deposition tilt θ from normal [°]  (recipe: 60)
    manhattan_delta: float = 15.0 # in-plane offset δ [°]              (recipe: 15 / 25)
    manhattan_h:     float = 1800.0  # upper imaging-resist thickness h [nm] (recipe: ~1.8 µm)

    # ── Mode ──────────────────────────────────────────────────
    mode: str = "Dolan bridge"    # "Dolan bridge" | "Manhattan"

    # ── Deposition stack ──────────────────────────────────────
    # "Bilayer"  : evap1 → oxidation → evap2  (the classic setup).
    # "Trilayer" : evap1(Nb)→evap2(Al)→oxidation→evap3(Al)→evap4(Nb).
    #   Electrode 1 (bottom) = Nb(evap1) + Al(evap2) at the Evaporation-1 angle;
    #   Electrode 2 (top)    = Al(evap3) + Nb(evap4) at the Evaporation-2 angle.
    #   evap1 reuses (angle1, phi1); evap3 reuses (angle2, phi2).  The two upper
    #   sublayers (evap2, evap4) default to their electrode angle but are free.
    stack: str = "Bilayer"        # "Bilayer" | "Trilayer"
    tri_t1: float =  80.0         # Nb (electrode-1 lower) thickness [nm]
    tri_t2: float =  10.0         # Al (electrode-1 upper) thickness [nm]
    tri_t3: float =  10.0         # Al (electrode-2 lower) thickness [nm]
    tri_t4: float = 150.0         # Nb (electrode-2 upper) thickness [nm]
    tri_angle2: float = -24.0     # evap2 (Al, electrode-1 upper) tilt [°]
    tri_phi2:   float =   0.0     # evap2 azimuth [°]
    tri_angle4: float =  24.0     # evap4 (Nb, electrode-2 upper) tilt [°]
    tri_phi4:   float =   0.0     # evap4 azimuth [°]
    # Opt-in: a prior evaporation's resist-sidewall coating shadows later
    # evaporations (narrows their effective opening — Jpn. J. Appl. Phys. aca256).
    sidewall: bool = False

    @property
    def t_resist(self) -> float:
        return self.t_pmma + self.t_mma

    @property
    def t_gap(self) -> float:
        """Height of the suspended bridge above the substrate.

        The bridge underside sits at the top of the MMA layer (z = t_mma), so
        the under-bridge junction overlap is governed by the MMA height — NOT
        the full resist height.  Standard Dolan overlap:  2·t_mma·tanθ − bridge_len.
        """
        return self.t_mma

    # Dolan: x-direction opening half-widths
    @property
    def pmma_half_x(self) -> float:
        """PMMA half-gap in x (no undercut) = bridge_len/2."""
        return self.bridge_len / 2

    @property
    def mma_half_x(self) -> float:
        """MMA half-gap in x (with undercut)."""
        return self.bridge_len / 2 + self.undercut

    # Dolan: y-direction opening half-widths (side trench)
    @property
    def pmma_half_y(self) -> float:
        """PMMA half-width of side trench in y = bridge_w/2."""
        return self.bridge_w / 2

    @property
    def mma_half_y(self) -> float:
        """MMA half-width of side trench in y (with undercut)."""
        return self.bridge_w / 2 + self.undercut

    # backward compat
    @property
    def bridge_width(self) -> float:
        return self.bridge_w

    @property
    def uc_half(self) -> float:
        return self.mma_half_x


def shadow_vector(t_resist: float, theta_deg: float, phi_deg: float):
    """
    Shadow offset (sx, sy) on wafer surface.
    sx = t · tan(θ) · cos(φ)
    sy = t · tan(θ) · sin(φ)
    """
    t   = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    s   = t_resist * np.tan(t)
    return s * np.cos(phi), s * np.sin(phi)


def shadow_offset(t_resist: float, angle_deg: float, phi_deg: float = 0.0) -> float:
    sx, _ = shadow_vector(t_resist, angle_deg, phi_deg)
    return sx


# ════════════════════════════════════════════════════════════════
# Plassys point-source / tilted-wafer geometry
# ════════════════════════════════════════════════════════════════
# A real oblique evaporator has a point source at a FIXED lab location and
# tilts/rotates the WAFER so the beam meets the wafer normal at the nominal
# (θ, φ) AT THE WAFER CENTRE.  Because the source is at finite throw distance L,
# a device sitting off-centre on the wafer sees the beam arrive at a slightly
# different local angle (θ′, φ′) relative to the tilted wafer normal — which
# shifts each electrode's shadow offset and so makes the junction area drift
# with wafer position.  These helpers compute that localised (θ′, φ′); the voxel
# engine itself is unchanged (it just receives the localised angles).


def evap_beams(p: "ProcessParams"):
    """Active evaporations as (label, theta_attr, phi_attr, theta_nom, phi_nom).

    Mirrors exactly which (θ, φ) fields ``deposition3d.simulate`` feeds to
    ``beam_direction`` — bilayer/Manhattan use (angle1, phi1) & (angle2, phi2);
    trilayer additionally uses (tri_angle2, tri_phi2) & (tri_angle4, tri_phi4).
    """
    if getattr(p, "stack", "Bilayer") == "Trilayer":
        return [("evap1 Nb", "angle1",     "phi1",     p.angle1,     p.phi1),
                ("evap2 Al", "tri_angle2", "tri_phi2", p.tri_angle2, p.tri_phi2),
                ("evap3 Al", "angle2",     "phi2",     p.angle2,     p.phi2),
                ("evap4 Nb", "tri_angle4", "tri_phi4", p.tri_angle4, p.tri_phi4)]
    return [("evap1", "angle1", "phi1", p.angle1, p.phi1),
            ("evap2", "angle2", "phi2", p.angle2, p.phi2)]


def _wafer_rot(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Wafer→lab rotation ``R = Ry(θ)·Rz(−φ)`` (right-handed lab-axis rotations).

    Columns are the wafer in-plane X, Y axes and the wafer normal, in lab coords.
    Chosen so the fixed vertical beam ``b0 = (0,0,−1)`` expressed in the wafer
    frame equals ``beam_direction(θ, φ)`` at the wafer centre:
        Rᵀ·b0 = (sinθcosφ, sinθsinφ, −cosθ).
    Physically: a fixed tilt axis (lab-y, normal leans toward lab +x by θ) plus a
    wafer spin φ about its own normal that sets the azimuth — the Plassys stage.
    """
    th = np.radians(theta_deg); ph = np.radians(phi_deg)
    Ry = np.array([[ np.cos(th), 0.0, np.sin(th)],
                   [        0.0, 1.0,        0.0],
                   [-np.sin(th), 0.0, np.cos(th)]])
    Rzm = np.array([[ np.cos(ph), np.sin(ph), 0.0],   # Rz(−φ)
                    [-np.sin(ph), np.cos(ph), 0.0],
                    [        0.0,        0.0, 1.0]])
    return Ry @ Rzm


def wafer_local_angles(theta_deg, phi_deg, X, Y, L, S0=None):
    """Local (θ′, φ′) [deg] a device at wafer-frame (X, Y) sees under the Plassys
    fixed-source / tilted-wafer model.

    Source at lab origin; wafer centre at (0, 0, −L) facing the source; wafer
    tilted by ``R = Ry(θ)·Rz(−φ)``.  ``X``, ``Y``, ``L`` share length units (mm).
    Optional ``S0`` = (x, y, z) lab source offset [mm] (default origin) models a
    displaced / finite source — the beam direction is taken from ``S0`` to the
    device instead of from the origin.
    Vectorised over array ``X``/``Y``.  Returns the nominal-equivalent angle at
    X = Y = 0 (for a negative nominal θ the physically-identical positive-θ /
    flipped-φ pair is returned; the engine is angle-convention-agnostic).
    """
    R = _wafer_rot(theta_deg, phi_deg)
    eX, eY = R[:, 0], R[:, 1]
    C = np.array([0.0, 0.0, -float(L)])
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    P = C + X[..., None] * eX + Y[..., None] * eY        # lab positions (..., 3)
    if S0 is not None:                                   # finite / displaced src
        P = P - np.asarray(S0, float)                    # beam from S0 → device
    d = P / np.linalg.norm(P, axis=-1, keepdims=True)     # unit beam direction
    dloc = d @ R                                          # == Rᵀ·d  (per row)
    th = np.degrees(np.arccos(np.clip(-dloc[..., 2], -1.0, 1.0)))
    ph = np.degrees(np.arctan2(dloc[..., 1], dloc[..., 0]))
    return th, ph


def wafer_source_dist(theta_deg, phi_deg, X, Y, L):
    """Source(origin)→device distance [mm] for a device at wafer-frame (X, Y) on a
    wafer tilted to (theta, phi) at throw distance L.

    ``P = C + X·eX + Y·eY`` (C = wafer centre at (0,0,−L)); the returned ``|P|`` is
    the beam path length, which varies with wafer position and — because the
    in-plane offset tilts with the wafer — per evaporation.  Vectorised over X/Y.
    """
    R = _wafer_rot(theta_deg, phi_deg)
    eX, eY = R[:, 0], R[:, 1]
    C = np.array([0.0, 0.0, -float(L)])
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    P = C + X[..., None] * eX + Y[..., None] * eY
    return np.linalg.norm(P, axis=-1)


def wafer_params(p: "ProcessParams", X, Y, L) -> "ProcessParams":
    """``copy.copy(p)`` with every active evaporation's (θ, φ) replaced by its
    local angle for scalar wafer position (X, Y).

    Only the beam-angle fields are touched, so the single-JJ ProcessParams and
    every other geometry/resist parameter are left untouched.
    """
    q = copy.copy(p)
    for _lbl, ta, pa, th, ph in evap_beams(p):
        lth, lph = wafer_local_angles(th, ph, float(X), float(Y), L)
        setattr(q, ta, float(lth)); setattr(q, pa, float(lph))
    return q


def sample_beam_cloud(pattern, size, n, rng):
    """``n``×2 array of source-plane offsets ``(dx, dy)`` [mm] drawn from the e-beam
    raster intensity ``pattern`` (how the source spread is shaped on the target).

    All patterns are azimuthally isotropic.  ``size`` meaning: ``gaussian`` → σ
    (r.m.s.); ``uniform`` / ``rotline`` → disk diameter.  ``point`` → zeros (ideal
    point source).  ``rotline`` is a line spot rotated uniformly (the standard recipe)
    → an area density ∝ 1/ρ, i.e. *uniform in radius*.  ``rng`` is a numpy
    ``Generator``; any unknown pattern degrades to a point source.
    """
    n = int(n); s = float(size)
    if s <= 0 or pattern == "point":
        return np.zeros((n, 2))
    if pattern == "gaussian":                            # isotropic 2-D Gaussian, σ=s
        return rng.normal(0.0, s, size=(n, 2))
    if pattern == "uniform":                             # area-uniform disk, dia = s
        r = (s / 2.0) * np.sqrt(rng.random(n)); ph = rng.random(n) * 2.0 * np.pi
        return np.column_stack([r * np.cos(ph), r * np.sin(ph)])
    if pattern == "rotline":                             # rotating line → 1/ρ disk
        t = (rng.random(n) - 0.5) * s                    # point on the line, |t| ≤ s/2
        ph = rng.random(n) * 2.0 * np.pi                 # uniform rotation
        return np.column_stack([t * np.cos(ph), t * np.sin(ph)])
    return np.zeros((n, 2))


def wafer_params_source(p: "ProcessParams", X, Y, L, pattern, size, rng):
    """One Monte-Carlo draw of ``wafer_params`` for a finite source whose spatial
    spread follows ``pattern`` (see :func:`sample_beam_cloud`).

    Each evaporation independently draws a transverse source offset (a separate
    physical deposition), so its local incidence angle is perturbed accordingly —
    computed exactly via the displaced-source path of ``wafer_local_angles`` (no
    small-angle approximation).  Returns a ProcessParams copy with the perturbed
    (θ, φ); only beam-angle fields are touched, so the single-JJ params and the
    engine stay untouched.
    """
    q = copy.copy(p)
    beams = evap_beams(p)
    offs = sample_beam_cloud(pattern, size, len(beams), rng)
    for (_lbl, ta, pa, th, ph), (dx, dy) in zip(beams, offs):
        lth, lph = wafer_local_angles(th, ph, float(X), float(Y), L,
                                      S0=np.array([dx, dy, 0.0]))
        setattr(q, ta, float(lth)); setattr(q, pa, float(lph))
    return q


def wafer_params_gaussian(p: "ProcessParams", X, Y, L, sigma_src, rng):
    """Back-compat wrapper: a Gaussian-pattern draw of :func:`wafer_params_source`
    (finite source of r.m.s. size ``sigma_src`` [mm])."""
    return wafer_params_source(p, X, Y, L, "gaussian", sigma_src, rng)
