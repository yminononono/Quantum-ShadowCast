"""
junction_area.py  (v6)
======================
Corrected Dolan bridge geometry and Manhattan cross with independent phi.

═══════════════════════════════════════════════════════════════════
DOLAN BRIDGE — Correct Physics
═══════════════════════════════════════════════════════════════════

Structure (top view, x = evaporation direction):

         gap_R (right gap)      gap_L (left gap)
      ←──────────────→        ←──────────────→
  ┌────────────┐   ╔══════════╗   ┌────────────┐
  │ Left PMMA  │   ║  bridge  ║   │ Right PMMA │
  │ electrode  │   ╚══════════╝   │ electrode  │
  └────────────┘                  └────────────┘
  x: -(L/2+u) -(L/2)    0    +(L/2)  +(L/2+u)
                MMA gap ←──────────→ MMA gap

  The bridge spans x ∈ [-L/2, +L/2].
  On each side of the bridge is a gap:
    Right gap: x ∈ [+L/2, +L/2+u]  (MMA undercut only)
    Left  gap: x ∈ [-L/2-u, -L/2]  (MMA undercut only)

Evaporation 1 (θ₁ < 0, e.g. -25°):
  Beam direction:  right-to-left  (dx < 0, enters from the RIGHT)
  Shadow offset:   sx1 = t·tan(θ₁) < 0   (beam shifts deposit LEFT)
  Beam enters through RIGHT gap.
  The bridge LEFT edge (-L/2) does NOT block it directly;
  the bridge RIGHT edge (+L/2) acts as the shadow-casting overhang.

  On the wafer floor (under bridge):
    The beam enters at x = +L/2 (right edge of bridge) and tilts left.
    It reaches the floor at:  x_floor = +L/2 + sx1   (< +L/2)
    The beam also enters at x = +L/2 + u (right MMA wall), floor at:
      x_floor = (+L/2+u) + sx1
    But we clip to the MMA opening.

    Evap1 floor deposit: x ∈ [+L/2 + sx1,  +L/2 + u]
      (i.e. the region under the RIGHT gap + a bit under bridge)
      Clipped to MMA gap: x ∈ [max(-mma_hx, +L/2+sx1), +mma_hx]

    IMPORTANT: Since sx1 < 0, the deposit extends to the LEFT of +L/2,
    reaching under the bridge by |sx1| amount.

Evaporation 2 (θ₂ > 0, e.g. +25°):
  Beam direction:  left-to-right  (dx > 0, enters from the LEFT)
  sx2 = t·tan(θ₂) > 0
  Evap2 floor deposit: x ∈ [-mma_hx, -L/2 + sx2]
    (left gap + under bridge by sx2 amount)

Junction overlap (where both deposits meet UNDER the bridge):
  x ∈ [-L/2 + sx2,  +L/2 + sx1]
  overlap_x = (+L/2 + sx1) - (-L/2 + sx2) = L + sx1 - sx2

  For symmetric θ (sx1 = -|sx|, sx2 = +|sx|):
    overlap_x = L - 2|sx| = L - 2·t·tan|θ|
  Condition for junction: overlap_x > 0  →  L > 2·t·tan|θ|

y-direction (bridge width, φ=0):
  Both deposits span y ∈ [-mma_hy, +mma_hy] (the side trench opening).
  With φ≠0: the deposit stripe shifts in y by sy, reducing overlap_y.

═══════════════════════════════════════════════════════════════════
MANHATTAN CROSS — Correct Physics
═══════════════════════════════════════════════════════════════════

Cross-shaped resist opening:
  x-arm: y ∈ [-wx/2, +wx/2],  x extends to ±infinity (within wafer)
  y-arm: x ∈ [-wy/2, +wy/2],  y extends to ±infinity

Evap1 (φ₁, θ₁): deposit stripe offset by (sx1, sy1)
  Floor deposit constrained by x-arm opening in y:
    x: full width (electrode width, say ±arm_len)
    y: [-wx/2 - uc + sy1, +wx/2 + uc + sy1]  ← shifted by sy1 in y
       but CLIPPED to y-arm opening: y ∈ [-wy/2 - uc, +wy/2 + uc]

  Actually the deposit is a stripe along the evaporation direction,
  clipped by the resist walls:
    The beam travels at angle (θ,φ). The opening in PMMA is cross-shaped.
    Metal lands where the beam can reach the floor WITHOUT being blocked.

  For φ₁≈0 (beam in x-direction): deposit stripe is along x.
    y extent: [-wx/2 - uc + sy1, +wx/2 + uc + sy1]  (MMA y-arm opening, shifted)
    x extent: the full arm length (PMMA allows x for all y in opening)

  For φ₂≈90° (beam in y-direction): deposit stripe is along y.
    x extent: [-wy/2 - uc + sx2, +wy/2 + uc + sx2]  (shifted by sx2)
    y extent: full arm length

  The deposit in EACH case is a cross-shaped region:
    (stripe along beam direction) ∩ (cross-shaped opening)
    = the arm that is parallel to the beam direction

  So Evap1 (φ≈0) deposits a stripe in the x-arm: full x-arm, y ∈ MMA x-arm opening.
  Evap2 (φ≈90°) deposits a stripe in the y-arm: full y-arm, x ∈ MMA y-arm opening.
  Junction = cross-center overlap where both stripes meet.
"""

import numpy as np
from process_engine import ProcessParams, shadow_vector

JC_AL_kA_cm2 = 10.0


def compute_junction_area(params: ProcessParams) -> dict:
    p = params
    if p.mode == "Dolan bridge":
        # Under-bridge tongue is shadowed by the bridge underside at z=t_mma,
        # so the junction overlap is governed by t_mma (the air-gap height),
        # not the full PMMA+MMA resist stack.
        h = p.t_gap
        sx1, sy1 = shadow_vector(h, p.angle1, p.phi1)
        sx2, sy2 = shadow_vector(h, p.angle2, p.phi2)
        return _dolan(p, sx1, sy1, sx2, sy2)
    else:
        t = p.t_resist
        sx1, sy1 = shadow_vector(t, p.angle1, p.phi1)
        sx2, sy2 = shadow_vector(t, p.angle2, p.phi2)
        return _manhattan_cross(p, sx1, sy1, sx2, sy2)


def _dolan(p, sx1, sy1, sx2, sy2):
    L      = p.bridge_len
    W      = p.bridge_w
    u      = p.undercut
    mma_hx = L/2 + u    # MMA x half-width (full gap = L + 2u)
    mma_hy = W/2 + u    # MMA y half-width (side trench + undercut)

    # ── Evap1: beam from RIGHT (θ₁<0, sx1<0) ─────────────────────
    # Beam enters right gap, bridge right edge (+L/2) casts shadow leftward.
    # Floor deposit spans from where bridge shadow ends to right MMA wall.
    #   right MMA wall: x = +mma_hx
    #   bridge shadow edge on floor: x = +L/2 + sx1   (sx1<0 → inside bridge)
    # But beam also illuminates the right gap itself (x ∈ [+L/2, +mma_hx]):
    # all of that gap is lit + under-bridge region up to shadow edge.
    ev1_x_lo = max(-mma_hx, L/2 + sx1)   # shadow edge, clipped to MMA
    ev1_x_hi = mma_hx                      # right MMA wall

    # ── Evap2: beam from LEFT (θ₂>0, sx2>0) ──────────────────────
    ev2_x_lo = -mma_hx                     # left MMA wall
    ev2_x_hi = min(mma_hx, -L/2 + sx2)   # shadow edge, clipped

    # ── Junction overlap (under bridge center) ────────────────────
    ov_x_lo  = max(ev1_x_lo, ev2_x_lo)
    ov_x_hi  = min(ev1_x_hi, ev2_x_hi)
    overlap_x = ov_x_hi - ov_x_lo   # = L + sx1 - sx2 (when both in range)

    # ── y-direction ───────────────────────────────────────────────
    y1_lo, y1_hi = -mma_hy + sy1,  mma_hy + sy1
    y2_lo, y2_hi = -mma_hy + sy2,  mma_hy + sy2
    overlap_y = min(y1_hi, y2_hi) - max(y1_lo, y2_lo)

    jrect = []
    if overlap_x > 0 and overlap_y > 0:
        jrect = [(ov_x_lo, max(y1_lo,y2_lo)),
                 (ov_x_hi, max(y1_lo,y2_lo)),
                 (ov_x_hi, min(y1_hi,y2_hi)),
                 (ov_x_lo, min(y1_hi,y2_hi))]

    area_nm2 = max(overlap_x, 0) * max(overlap_y, 0)
    ic_uA    = JC_AL_kA_cm2 * 1e3 * (area_nm2 * 1e-14) * 1e6
    tilt     = np.degrees(np.arctan2(sy2-sy1, max(overlap_x, 1e-9)))

    return dict(
        sx1=sx1, sy1=sy1, sx2=sx2, sy2=sy2,
        ev1_x_lo=ev1_x_lo, ev1_x_hi=ev1_x_hi,
        ev2_x_lo=ev2_x_lo, ev2_x_hi=ev2_x_hi,
        ov_x_lo=ov_x_lo,   ov_x_hi=ov_x_hi,
        y1_lo=y1_lo, y1_hi=y1_hi,
        y2_lo=y2_lo, y2_hi=y2_hi,
        overlap_x_nm=overlap_x, overlap_y_nm=overlap_y,
        dep1_right_nm=ev1_x_hi, dep2_left_nm=ev2_x_lo,
        area_nm2=area_nm2, junction_tilt_deg=tilt,
        junction_rect=jrect, ic_estimate_uA=ic_uA,
        uc_half_nm=mma_hx, mma_hx=mma_hx, mma_hy=mma_hy,
        shadow1_nm=sx1, shadow2_nm=sx2, overlap_nm=overlap_x,
        mode="Dolan bridge",
    )


def _manhattan_cross(p, sx1, sy1, sx2, sy2):
    """
    Manhattan-style double-oblique evaporation (arxiv:2605.19590, App. A).

    Two perpendicular resist line openings cross at the origin:
      - x-running arm: long in x, designed opening width w_open_x in y.
      - y-running arm: long in y, designed opening width w_open_y in x.

    Each electrode line is deposited by an oblique beam tilted by θ from the
    substrate normal and offset in-plane by δ from the line direction. The
    upper resist (thickness h) shadows the opening, narrowing the deposited
    line to (Eq. A6):

        w_narrow = w_open − h · sin(δ) / tan(θ)

    The Josephson junction is the crossing of the two narrowed lines:
        area = w_narrow_x · w_narrow_y
    (idealized symmetric model — neglects 1st-layer metal accumulation that
     makes the two real linewidths asymmetric.)
    """
    theta = abs(p.manhattan_theta)
    delta = abs(p.manhattan_delta)
    h     = p.manhattan_h
    wx    = p.manhattan_wx          # designed opening of x-running arm (y-width)
    wy    = p.manhattan_wy          # designed opening of y-running arm (x-width)

    tan_t  = np.tan(np.radians(theta))
    shrink = h * np.sin(np.radians(delta)) / tan_t if tan_t > 1e-9 else 0.0  # Eq. A6

    wnx = wx - shrink               # narrowed x-arm width (junction y-extent)
    wny = wy - shrink               # narrowed y-arm width (junction x-extent)

    overlap_x = max(wny, 0.0)       # junction extent in x  (set by y-arm)
    overlap_y = max(wnx, 0.0)       # junction extent in y  (set by x-arm)
    area_nm2  = overlap_x * overlap_y
    ic_uA     = JC_AL_kA_cm2 * 1e3 * (area_nm2 * 1e-14) * 1e6

    hx, hy = overlap_x / 2, overlap_y / 2
    jrect = []
    if overlap_x > 0 and overlap_y > 0:
        jrect = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]

    arm_len = 3000
    # Legacy keys so cross_section.py renders a sensible junction crossing:
    #   ev1 = x-running arm (spans full x), ev2 = y-running arm (narrowed in x).
    ev1_x_lo, ev1_x_hi = -arm_len, arm_len
    ev2_x_lo, ev2_x_hi = -hx, hx

    return dict(
        sx1=sx1, sy1=sy1, sx2=sx2, sy2=sy2,
        # double-oblique narrowing details
        shrink_nm=shrink, wnarrow_x_nm=max(wnx, 0.0), wnarrow_y_nm=max(wny, 0.0),
        theta_deg=theta, delta_deg=delta, h_nm=h,
        wopen_x_nm=wx, wopen_y_nm=wy, arm_len=arm_len,
        # junction
        overlap_x_nm=overlap_x, overlap_y_nm=overlap_y,
        area_nm2=area_nm2, junction_tilt_deg=0.0,
        junction_rect=jrect, ic_estimate_uA=ic_uA,
        jx_lo=-hx, jx_hi=hx, jy_lo=-hy, jy_hi=hy,
        ov_x_lo=-hx, ov_x_hi=hx,
        # legacy keys for cross_section.py
        ev1_x_lo=ev1_x_lo, ev1_x_hi=ev1_x_hi,
        ev2_x_lo=ev2_x_lo, ev2_x_hi=ev2_x_hi,
        dep1_right_nm=ev1_x_hi, dep2_left_nm=ev2_x_lo,
        uc_half_nm=wy/2, mma_hx=wy/2, mma_hy=wx/2,
        shadow1_nm=sx1, shadow2_nm=sx2, overlap_nm=overlap_x,
        mode="Manhattan",
    )
