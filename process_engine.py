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
import numpy as np


@dataclass
class ProcessParams:
    """All process parameters (nm / degrees)."""

    # ── Bilayer resist ─────────────────────────────────────────
    # Recipe: arxiv:2101.01453 — PMMA A-4 ~250 nm on MMA EL-13 ~900 nm.
    t_pmma:   float = 250.0   # PMMA top layer [nm]  (no undercut)
    t_mma:    float = 900.0   # MMA  bot layer [nm]  (undercut sublayer)
    undercut: float = 150.0   # MMA one-sided undercut [nm]
    # Suspended-bridge air-gap height (shadow-defining).  Independent control;
    # defaults to the MMA thickness but can be set freely.
    bridge_gap: float = 900.0  # bridge underside height above substrate [nm]

    # ── Evaporation 1 ─────────────────────────────────────────
    # Uniaxial tilt ±24° (same azimuth φ, opposite polar θ); 30 nm Al each.
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

    @property
    def t_resist(self) -> float:
        return self.t_pmma + self.t_mma

    @property
    def t_gap(self) -> float:
        """Height of the suspended bridge above the substrate.

        The metal tongue that reaches under the bridge is shadowed by the
        bridge bottom edge, so the under-bridge junction overlap is governed by
        this gap — NOT the full resist height.
        Standard Dolan overlap:  2·bridge_gap·tanθ − bridge_len.
        """
        return self.bridge_gap

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
