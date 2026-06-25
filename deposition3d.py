"""
deposition3d.py
===============
Physical shadow-evaporation engine.

Instead of hand-drawing where metal "should" be, this module builds the actual
3-D resist geometry (substrate + MMA + PMMA + Dolan bridge / Manhattan cross) as
axis-aligned boxes and traces the parallel evaporation beam with ray ↔ box
occlusion tests.  Metal is deposited on every surface that

  1. faces the incoming beam (its beam-forward neighbour cell is solid), and
  2. is illuminated  (the ray from that cell toward the source is not blocked).

The result is rasterised into a labelled voxel grid so that an arbitrary cross
section (x–z at a given y, or y–z at a given x) can be sliced out, and the
junction overlap / area measured directly from the voxels.

All occluders are axis-aligned boxes, so occlusion is done with a vectorised
slab ray–AABB test in NumPy — no rtree/embree needed.
"""

from dataclasses import dataclass
import numpy as np

from process_engine import ProcessParams, wafer_local_angles, sample_beam_cloud


# ── voxel labels ────────────────────────────────────────────────
EMPTY = 0
RESIST = 1
SUBSTRATE = 2

MAX_CELLS_PER_AXIS = 140   # resolution cap (keeps the grid tractable)


# ════════════════════════════════════════════════════════════════
# Geometry  →  list of opaque axis-aligned boxes
# ════════════════════════════════════════════════════════════════

def _box(x0, x1, y0, y1, z0, z1):
    return (min(x0, x1), max(x0, x1),
            min(y0, y1), max(y0, y1),
            min(z0, z1), max(z0, z1))


def _solid_from_openings(openings, R, z0, z1):
    """Exact complement of axis-aligned opening rectangles within [-R,R]².

    `openings` is a list of (x0, x1, y0, y1).  Returns solid boxes z∈[z0,z1]
    covering everywhere in the domain NOT inside any opening, by partitioning
    on the union of all rectangle edges (handles L-corners, crosses, etc.).
    """
    xb = sorted(set([-R, R] + [v for o in openings for v in (o[0], o[1])]))
    yb = sorted(set([-R, R] + [v for o in openings for v in (o[2], o[3])]))
    boxes = []
    for i in range(len(xb) - 1):
        for j in range(len(yb) - 1):
            cx = 0.5 * (xb[i] + xb[i + 1]); cy = 0.5 * (yb[j] + yb[j + 1])
            in_open = any(o[0] <= cx <= o[1] and o[2] <= cy <= o[3]
                          for o in openings)
            if not in_open:
                boxes.append(_box(xb[i], xb[i + 1], yb[j], yb[j + 1], z0, z1))
    return boxes


def _frame_boxes(hx, hy, z0, z1, R):
    """A solid layer of half-domain R with a rectangular hole (|x|<hx,|y|<hy).

    Returned as 4 boxes forming the frame around the hole.
    """
    return [
        _box(-R, -hx, -R,  R, z0, z1),   # left
        _box(hx,  R, -R,  R, z0, z1),    # right
        _box(-hx, hx,  hy,  R, z0, z1),  # top    (between the side walls)
        _box(-hx, hx, -R, -hy, z0, z1),  # bottom
    ]


def _cross_boxes(A, B, R, z0, z1):
    """Exact 8-box complement of two centred, perpendicular strips sharing the
    same reach (Manhattan's resist cross): A=(-arm,arm,-hyA,hyA) ∪
    B=(-hxB,hxB,-arm,arm).  Equivalent to _solid_from_openings([A, B], ...)
    but with 8 boxes instead of ~20."""
    arm = A[1]; hyA = A[3]; hxB = B[1]
    return [
        _box(-R, R, arm, R, z0, z1),          # top strip
        _box(-R, R, -R, -arm, z0, z1),        # bottom strip
        _box(-R, -arm, -arm, arm, z0, z1),    # left strip
        _box(arm, R, -arm, arm, z0, z1),      # right strip
        _box(-arm, -hxB, hyA, arm, z0, z1),   # 4 corners
        _box(hxB, arm, hyA, arm, z0, z1),
        _box(-arm, -hxB, -arm, -hyA, z0, z1),
        _box(hxB, arm, -arm, -hyA, z0, z1),
    ]


def _rect_boxes(openings, R, z0, z1):
    """Cheaper exact equivalents of _solid_from_openings for the two opening
    shapes this module ever builds (Dolan's single centred rectangle;
    Manhattan's two centred perpendicular strips sharing the same reach),
    falling back to the general method for anything else."""
    if len(openings) == 1:
        x0, x1, y0, y1 = openings[0]
        hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        if abs(x0 + x1) < 1e-6 and abs(y0 + y1) < 1e-6:
            return _frame_boxes(hx, hy, z0, z1, R)
    elif len(openings) == 2:
        rects = openings
        centred = all(abs(x0 + x1) < 1e-6 and abs(y0 + y1) < 1e-6
                      for x0, x1, y0, y1 in rects)
        if centred:
            (x0a, x1a, y0a, y1a) = rects[0]
            (x0b, x1b, y0b, y1b) = rects[1]
            if x1a > y1a and y1b > x1b and abs(x1a - y1b) < 1e-6:
                return _cross_boxes(rects[0], rects[1], R, z0, z1)
            if y1a > x1a and x1b > y1b and abs(y1a - x1b) < 1e-6:
                return _cross_boxes(rects[1], rects[0], R, z0, z1)
    return _solid_from_openings(openings, R, z0, z1)


def _e_of_z(z, z0, z1, rr, round_top, round_bot):
    """Continuous quarter-circle wall-retreat e(z) for a resist layer [z0,z1]
    rounded by radius `rr` at its top (z1) and/or bottom (z0) face: 0 ⇒ sharp
    wall, `rr` ⇒ fully retreated at the rounded face.  `z` may be an array.
    This is the exact (K→∞) continuous limit of the old K-slab fillet
    approximation, used by `_label_solid`'s point test and (via
    ``voxel_view``) the cross-section curve overlay."""
    z = np.asarray(z, float)
    h_top = (z - (z1 - rr)) if round_top else np.full_like(z, -1.0)
    h_bot = ((z0 + rr) - z) if round_bot else np.full_like(z, -1.0)
    h = np.clip(np.maximum(h_top, h_bot), 0.0, rr)
    active = np.maximum(h_top, h_bot) >= 0
    return np.where(active, rr - np.sqrt(np.maximum(rr * rr - h * h, 0.0)), 0.0)


@dataclass
class RoundedRect:
    """One centred rectangular resist opening whose walls retreat by a shared
    quarter-circle e(z) near the top and/or bottom z-face (Dolan's rounded
    trench) — the analytic, continuous replacement for the old K-slab fillet
    box stack.  Solid = complement of {|x|<hx+e(z), |y|<hy+e(z)} within
    [z0,z1]."""
    hx: float; hy: float
    z0: float; z1: float
    rr: float
    round_top: bool
    round_bot: bool


@dataclass
class RoundedCross:
    """Manhattan's resist cross: two perpendicular RoundedRect strips sharing
    the same z0/z1/rr/round_top/round_bot (hence the same e(z)) — solid =
    intersection of each strip's own complement.  `arm` is the shared far
    reach of both strips; `hy_a`/`hx_b` are strip A's y-half-width and strip
    B's x-half-width.  No separate corner-rounding term is needed: rounding
    only ever grows each strip's own transverse half-width, never `arm`, so
    the two rounded surfaces are already consistent where they meet."""
    arm: float; hy_a: float; hx_b: float
    z0: float; z1: float
    rr: float
    round_top: bool
    round_bot: bool


def _t_range_for_z(o, d, za, zb, eps):
    """(N,) ta, tb, valid — the ray's t-sub-range where z(t)∈[za,zb], t>eps.
    `tb` is +inf when the ray is z-parallel and inside the slab (the actual
    bound then comes from whichever x/y test uses this range, exactly as
    `_occluded`'s own slab test leaves an unconstrained axis unbounded)."""
    N = len(o)
    dz = float(d[2])
    if abs(dz) < 1e-12:
        valid = (o[:, 2] >= za) & (o[:, 2] <= zb)
        return np.full(N, eps), np.full(N, np.inf), valid
    ta_raw = (za - o[:, 2]) / dz
    tb_raw = (zb - o[:, 2]) / dz
    ta = np.maximum(np.minimum(ta_raw, tb_raw), eps)
    tb = np.maximum(ta_raw, tb_raw)
    return ta, tb, ta < tb


def _sharp_wall_hit(o, d, half, axis, sign, ta, tb, valid):
    """(N,) bool — does sign·coord_axis(t) exceed the flat `half`-width wall
    for some t∈[ta,tb]?"""
    oc, dc = sign * o[:, axis], sign * float(d[axis])
    if abs(dc) < 1e-12:
        return valid & (oc > half)
    t_cross = (half - oc) / dc
    lo, hi = (t_cross, tb) if dc > 0 else (ta, t_cross)
    lo, hi = np.maximum(lo, ta), np.minimum(hi, tb)
    return valid & (lo < hi)


def _rounded_wall_hit(o, d, half, rr, z_face, zsign, axis, sign, ta, tb, valid):
    """(N,) bool — the quadratic rounded-wall test (does sign·coord_axis(t)
    exceed the retreating `half`+e(z) boundary for some t∈[ta,tb]?).  Letting
    A(t)=half+rr-sign·coord_axis(t) and h(t) the (already clipped-range)
    distance into the fillet, the condition reduces to "A(t)<0" (past even
    the maximum retreat) OR "A(t)²+h(t)²<rr²" (inside the quarter-circle) —
    both linear/quadratic in t, solved directly instead of approximated by
    K discrete slabs.  `zsign`=+1 for a top-face rounding
    (h=z-(z_face-rr)), -1 for a bottom-face rounding (h=(z_face+rr)-z)."""
    N = len(o)
    if zsign > 0:
        h0 = o[:, 2] - z_face + rr; h1 = float(d[2])
    else:
        h0 = z_face + rr - o[:, 2]; h1 = -float(d[2])
    oc, dc = sign * o[:, axis], sign * float(d[axis])
    a0 = half + rr - oc; a1 = -dc
    hit = np.zeros(N, bool)
    if abs(a1) < 1e-12:
        hit |= valid & (a0 < 0)
    else:
        t_a0 = -a0 / a1
        if a1 > 0:
            hit |= valid & (ta < np.minimum(tb, t_a0))
        else:
            hit |= valid & (np.maximum(ta, t_a0) < tb)
    A = a1 * a1 + h1 * h1
    B = 2 * (a0 * a1 + h0 * h1)
    C = a0 * a0 + h0 * h0 - rr * rr
    if A > 1e-12:
        disc = B * B - 4 * A * C
        ok = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        r1 = (-B - sq) / (2 * A); r2 = (-B + sq) / (2 * A)
        qa, qb = np.minimum(r1, r2), np.maximum(r1, r2)
        lo, hi = np.maximum(qa, ta), np.minimum(qb, tb)
        hit |= valid & ok & (lo < hi)
    return hit


def _wall_hit(o, d, half, rr, z0, z1, round_top, round_bot, axis, sign, eps):
    """(N,) bool — OR of the sharp-middle / bottom-rounded / top-rounded
    sub-ranges for one wall of a RoundedRect/RoundedCross strip."""
    zlo = z0 + (rr if round_bot else 0.0)
    zhi = z1 - (rr if round_top else 0.0)
    hit = np.zeros(len(o), bool)
    if round_bot:
        ta, tb, valid = _t_range_for_z(o, d, z0, z0 + rr, eps)
        hit |= _rounded_wall_hit(o, d, half, rr, z0, -1, axis, sign, ta, tb, valid)
    if zhi > zlo:
        ta, tb, valid = _t_range_for_z(o, d, zlo, zhi, eps)
        hit |= _sharp_wall_hit(o, d, half, axis, sign, ta, tb, valid)
    if round_top:
        ta, tb, valid = _t_range_for_z(o, d, z1 - rr, z1, eps)
        hit |= _rounded_wall_hit(o, d, half, rr, z1, +1, axis, sign, ta, tb, valid)
    return hit


def _corner_sharp_hit(o, d, half_x, half_y, sx, sy, ta, tb, valid):
    """(N,) bool — AND of two flat half-space tests sharing one t-sub-range
    (the un-rounded middle of a RoundedCross corner)."""
    def _iv(axis, sign, half):
        oc, dc = sign * o[:, axis], sign * float(d[axis])
        if abs(dc) < 1e-12:
            return ta, tb, (oc > half)
        t_cross = (half - oc) / dc
        lo, hi = (t_cross, tb) if dc > 0 else (ta, t_cross)
        lo, hi = np.maximum(lo, ta), np.minimum(hi, tb)
        return lo, hi, (lo < hi)
    lox, hix, okx = _iv(0, sx, half_x)
    loy, hiy, oky = _iv(1, sy, half_y)
    lo, hi = np.maximum(lox, loy), np.minimum(hix, hiy)
    return valid & okx & oky & (lo < hi)


def _corner_rounded_hit(o, d, half_x, half_y, sx, sy, rr, z_face, zsign, ta, tb, valid):
    """(N,) bool — AND of two rounded-wall tests sharing the SAME e(z) (a
    RoundedCross corner).  Reduces to the single-wall quadratic test with
    A(t) replaced by max(Ax(t), Ay(t)); since that max is piecewise-linear
    (one kink where Ax(t)=Ay(t)), the t-range is split there and each half
    uses the single-wall formula with whichever side is locally larger."""
    N = len(o)
    if zsign > 0:
        h0 = o[:, 2] - z_face + rr; h1 = float(d[2])
    else:
        h0 = z_face + rr - o[:, 2]; h1 = -float(d[2])
    ocx, dcx = sx * o[:, 0], sx * float(d[0])
    ocy, dcy = sy * o[:, 1], sy * float(d[1])
    Ax0, Ax1 = half_x + rr - ocx, -dcx
    Ay0, Ay1 = half_y + rr - ocy, -dcy
    dA0, dA1 = Ax0 - Ay0, Ax1 - Ay1
    hit = np.zeros(N, bool)

    def _seg(sa, sb, use_x, ok):
        a0 = np.where(use_x, Ax0, Ay0)
        a1 = np.where(use_x, Ax1, Ay1)
        h_ = np.zeros(N, bool)
        zero_a1 = np.abs(a1) < 1e-12
        h_ |= ok & zero_a1 & (a0 < 0)
        nz = ok & ~zero_a1
        a1safe = np.where(a1 == 0, 1.0, a1)
        t_a0 = -a0 / a1safe
        h_ |= nz & (a1 > 0) & (sa < np.minimum(sb, t_a0))
        h_ |= nz & (a1 < 0) & (np.maximum(sa, t_a0) < sb)
        A = a1 * a1 + h1 * h1; B = 2 * (a0 * a1 + h0 * h1); C = a0 * a0 + h0 * h0 - rr * rr
        Asafe = np.where(A == 0, 1.0, A)
        disc = B * B - 4 * A * C
        okd = ok & (A > 1e-12) & (disc >= 0)
        sq = np.sqrt(np.maximum(disc, 0.0))
        r1 = (-B - sq) / (2 * Asafe); r2 = (-B + sq) / (2 * Asafe)
        qa, qb = np.minimum(r1, r2), np.maximum(r1, r2)
        lo, hi = np.maximum(qa, sa), np.minimum(qb, sb)
        h_ |= okd & (lo < hi)
        return h_

    if abs(dA1) < 1e-12:
        hit |= _seg(ta, tb, (Ax0 >= Ay0), valid & (ta < tb))
    else:
        t_kink = -dA0 / dA1
        use_x_left = (Ax0 + Ax1 * ta) >= (Ay0 + Ay1 * ta)
        kink_inside = (t_kink > ta) & (t_kink < tb)
        tk = np.clip(t_kink, ta, tb)
        use_x_2 = np.where(kink_inside, ~use_x_left, use_x_left)
        hit |= _seg(ta, tk, use_x_left, valid & (ta < tk))
        hit |= _seg(tk, tb, use_x_2, valid & (tk < tb))
    return hit


def _corner_hit(o, d, half_x, half_y, sx, sy, rr, z0, z1, round_top, round_bot, eps):
    """(N,) bool — OR of the sharp-middle / bottom-rounded / top-rounded
    sub-ranges for one corner of a RoundedCross."""
    zlo = z0 + (rr if round_bot else 0.0)
    zhi = z1 - (rr if round_top else 0.0)
    hit = np.zeros(len(o), bool)
    if round_bot:
        ta, tb, valid = _t_range_for_z(o, d, z0, z0 + rr, eps)
        hit |= _corner_rounded_hit(o, d, half_x, half_y, sx, sy, rr, z0, -1, ta, tb, valid)
    if zhi > zlo:
        ta, tb, valid = _t_range_for_z(o, d, zlo, zhi, eps)
        hit |= _corner_sharp_hit(o, d, half_x, half_y, sx, sy, ta, tb, valid)
    if round_top:
        ta, tb, valid = _t_range_for_z(o, d, z1 - rr, z1, eps)
        hit |= _corner_rounded_hit(o, d, half_x, half_y, sx, sy, rr, z1, +1, ta, tb, valid)
    return hit


def _occluded_rounded_rect(origins, d, rc, eps=1e-3):
    """Boolean (N,) — analytic equivalent of `_occluded` for one RoundedRect
    (Dolan's rounded trench): OR of its 4 walls."""
    o = np.asarray(origins, float)
    d = np.asarray(d, float)
    hit = _wall_hit(o, d, rc.hx, rc.rr, rc.z0, rc.z1, rc.round_top, rc.round_bot, 0, +1, eps)
    hit |= _wall_hit(o, d, rc.hx, rc.rr, rc.z0, rc.z1, rc.round_top, rc.round_bot, 0, -1, eps)
    hit |= _wall_hit(o, d, rc.hy, rc.rr, rc.z0, rc.z1, rc.round_top, rc.round_bot, 1, +1, eps)
    hit |= _wall_hit(o, d, rc.hy, rc.rr, rc.z0, rc.z1, rc.round_top, rc.round_bot, 1, -1, eps)
    return hit


def _occluded_rounded_cross(origins, d, rx, eps=1e-3):
    """Boolean (N,) — analytic equivalent of `_occluded` for one RoundedCross
    (Manhattan's rounded resist cross): OR of its 4 strip-end walls and 4
    corners (no special corner-rounding term needed — see RoundedCross)."""
    o = np.asarray(origins, float)
    d = np.asarray(d, float)
    args = (rx.rr, rx.z0, rx.z1, rx.round_top, rx.round_bot)
    hit = _wall_hit(o, d, rx.arm, *args, 1, +1, eps)
    hit |= _wall_hit(o, d, rx.arm, *args, 1, -1, eps)
    hit |= _wall_hit(o, d, rx.arm, *args, 0, -1, eps)
    hit |= _wall_hit(o, d, rx.arm, *args, 0, +1, eps)
    for sx in (+1, -1):
        for sy in (+1, -1):
            hit |= _corner_hit(o, d, rx.hx_b, rx.hy_a, sx, sy, *args, eps)
    return hit


def _layer_boxes(open_rects, R, z0, z1, r=0.0, round_top=False, round_bot=False):
    """Complement of `open_rects` in [-R,R]² over [z0,z1], with the opening
    optionally flared by a radius-`r` quarter-round at its top (z1) and/or bottom
    (z0) face — the resist lip / interface.  The wall retreats so the opening
    widens by up to r at the rounded face, following a quarter circle tangent to
    the wall and that face, approximated by `K` thin z-slabs.  `K` adapts to the
    radius (not the voxel grid), so the modelled fillet is a fixed, density-
    independent circle.  r<=0 ⇒ a plain layer.

    Legacy ``resist_round_method="voxel"`` path — kept alongside the analytic
    RoundedRect/RoundedCross model (`_occluded_rounded_rect`/`_cross`) as a
    slower but independently-implemented option; see `build_occluders`.
    """
    rr = float(max(0.0, r))
    if rr <= 0.0 or not (round_top or round_bot):
        return _rect_boxes(open_rects, R, z0, z1)
    faces = int(round_top) + int(round_bot)
    rr = min(rr, (z1 - z0) / faces)                        # keep the fillets apart
    K = int(np.clip(round(rr / 5.0), 10, 20))              # ~5 nm ledges; smooth circle

    def _exp(rects, e):
        return [(x0 - e, x1 + e, y0 - e, y1 + e) for (x0, x1, y0, y1) in rects]

    boxes = []
    zlo = z0 + (rr if round_bot else 0.0)
    zhi = z1 - (rr if round_top else 0.0)
    if zhi > zlo:                                           # nominal middle
        boxes += _rect_boxes(open_rects, R, zlo, zhi)
    if round_top:
        for k in range(K):
            za, zb = z1 - rr + k * rr / K, z1 - rr + (k + 1) * rr / K
            e = rr - np.sqrt(max(rr * rr - (0.5 * (za + zb) - (z1 - rr)) ** 2, 0.0))
            boxes += _rect_boxes(_exp(open_rects, e), R, za, zb)
    if round_bot:
        for k in range(K):
            za, zb = z0 + k * rr / K, z0 + (k + 1) * rr / K
            e = rr - np.sqrt(max(rr * rr - ((z0 + rr) - 0.5 * (za + zb)) ** 2, 0.0))
            boxes += _rect_boxes(_exp(open_rects, e), R, za, zb)
    return boxes


def build_occluders(p: ProcessParams):
    """Return (boxes, R, z_top, resist_h, z_split, rounded) for the given
    process.  ``rounded`` is a list of RoundedRect/RoundedCross — the
    analytic resist-rounding occluders OR'd in alongside ``boxes`` by
    `_occluded_any` — empty when ``resist_round<=0``."""
    if p.mode == "Dolan bridge":
        L = p.bridge_len                            # suspended bridge width (x)
        Wj = p.bridge_w                             # junction width (y)
        u = p.undercut
        gap, t_pmma = p.t_mma, p.t_pmma             # gap = MMA height = bridge underside
        z_top = gap + t_pmma

        # Horizontal opening between each bridge edge and the PMMA wall.  When
        # bridge_pmma_gap is 0 we auto-size it wide enough that the tilted beam
        # clears the wall and reaches under the bridge (bridge-limited junction,
        # overlap ≈ 2·gap·tanθ − L); otherwise the user sets it directly.
        tanmax = np.tan(np.radians(max(abs(p.angle1), abs(p.angle2))))
        auto_window = tanmax * z_top + L + 250.0
        window = p.bridge_pmma_gap if p.bridge_pmma_gap > 0 else auto_window
        trench_hx = L / 2 + window                  # PMMA trench half-width (x)
        ap_hy = Wj / 2                              # aperture half-width (y)
        R = trench_hx + u + 400.0

        rr = getattr(p, "resist_round", 0.0)
        boxes = []
        rounded = []
        if rr <= 0.0:
            # MMA (undercut sublayer) fills 0..gap; trench widened by undercut.
            boxes += _frame_boxes(trench_hx + u, ap_hy + u, 0.0, gap, R)
            # PMMA layer: open trench (no bridge yet), gap..z_top
            boxes += _frame_boxes(trench_hx, ap_hy, gap, z_top, R)
        else:
            # Rounded resist: only the PMMA (top) layer is filleted — its top lip
            # and its bottom face at the MMA interface.  The MMA (bottom) layer
            # stays sharp.  Two interchangeable models, selected by
            # `resist_round_method`: "analytic" (default, fast — RoundedRect /
            # `_occluded_rounded_rect`) or "voxel" (legacy K-slab box stack,
            # slower — `_layer_boxes`).
            boxes += _frame_boxes(trench_hx + u, ap_hy + u, 0.0, gap, R)
            if getattr(p, "resist_round_method", "analytic") == "voxel":
                boxes += _layer_boxes(
                    [(-trench_hx, trench_hx, -ap_hy, ap_hy)],
                    R, gap, z_top, rr, round_top=True, round_bot=True)
            else:
                rr_clamped = min(rr, (z_top - gap) / 2.0)   # keep top/bottom fillets apart
                rounded.append(RoundedRect(hx=trench_hx, hy=ap_hy, z0=gap, z1=z_top,
                                           rr=rr_clamped, round_top=True, round_bot=True))
        # Suspended bridge: narrow strip across the middle of the trench,
        # hanging over the air gap at z = gap.
        boxes.append(_box(-L / 2, L / 2, -ap_hy, ap_hy, gap, z_top))
        return boxes, R, z_top, gap, gap, rounded   # z_split = MMA/PMMA interface

    else:  # Manhattan double-oblique: two perpendicular resist lines crossing
        wA = p.manhattan_wx        # electrode A linewidth (A runs along x)
        wB = p.manhattan_wy        # electrode B linewidth (B runs along y)
        h = p.manhattan_h          # upper imaging-resist thickness
        u = p.undercut             # one-sided lateral undercut
        # Bilayer: a lower undercut sublayer (widened openings) carries the upper
        # imaging resist (nominal openings).  The undercut shelf lets metal lift
        # off cleanly, exactly like the Dolan MMA/PMMA stack.
        t_lo = p.t_mma                                # lower undercut sublayer = MMA
        z_top = t_lo + h
        # The electrode line must be long enough that the tilted beam can slide
        # all the way down to the floor in the crossing region: the beam clears
        # the resist top only after travelling z_top·tanθ horizontally, so the
        # opening has to extend at least that far past the junction.
        reach = z_top * np.tan(np.radians(abs(p.manhattan_theta)))
        arm = max(wA, wB) + reach + 400.0            # electrode arm half-length
        R = arm + u + 300.0

        # Cross (十字): both lines run fully through the centre, crossing at the
        # origin where the junction forms.
        A = (-arm, arm, -wA / 2, wA / 2)             # x-running line
        B = (-wB / 2, wB / 2, -arm, arm)             # y-running line
        # Undercut copies: transverse width widened by u on the lower sublayer.
        A_u = (-arm, arm, -wA / 2 - u, wA / 2 + u)
        B_u = (-wB / 2 - u, wB / 2 + u, -arm, arm)
        rr = getattr(p, "resist_round", 0.0)
        rounded = []
        if rr <= 0.0:
            boxes = _rect_boxes([A_u, B_u], R, 0.0, t_lo)   # lower (undercut)
            boxes += _rect_boxes([A, B], R, t_lo, z_top)    # upper (imaging)
        else:
            # Rounded resist: only the upper (imaging) layer is filleted — its top
            # lip and its bottom face at the interface.  The lower layer stays
            # sharp.  Two interchangeable models, selected by
            # `resist_round_method`: "analytic" (default, fast — RoundedCross /
            # `_occluded_rounded_cross`) or "voxel" (legacy K-slab box stack,
            # slower — `_layer_boxes`).
            boxes = _rect_boxes([A_u, B_u], R, 0.0, t_lo)
            if getattr(p, "resist_round_method", "analytic") == "voxel":
                boxes += _layer_boxes([A, B], R, t_lo, z_top, rr,
                                      round_top=True, round_bot=True)
            else:
                rr_clamped = min(rr, (z_top - t_lo) / 2.0)
                rounded.append(RoundedCross(arm=arm, hy_a=wA / 2, hx_b=wB / 2,
                                            z0=t_lo, z1=z_top, rr=rr_clamped,
                                            round_top=True, round_bot=True))
        return boxes, R, z_top, z_top, t_lo, rounded   # z_split = undercut/imaging interface


# ════════════════════════════════════════════════════════════════
# Ray ↔ AABB occlusion (vectorised slab method)
# ════════════════════════════════════════════════════════════════

def beam_direction(theta_deg, phi_deg):
    """Incoming beam travel direction (unit). θ from -z, azimuth φ in x–y.

    Horizontal component magnitude = sinθ, drop = cosθ, so a wall of height h
    shadows h·tanθ on the floor along (cosφ, sinφ): matches shadow_vector().
    """
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    return np.array([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     -np.cos(th)])


def _occluded(origins, d, boxes, eps=1e-3):
    """Boolean (N,) — does the ray origin + t·d (t>eps) hit any box?"""
    o = np.asarray(origins, float)
    N = len(o)
    hit = np.zeros(N, bool)
    dd = d.copy()
    dd[np.abs(dd) < 1e-12] = 1e-12          # avoid div-by-zero (slab handles parallel)
    inv = 1.0 / dd
    for (x0, x1, y0, y1, z0, z1) in boxes:
        lo = np.array([x0, y0, z0]); hi = np.array([x1, y1, z1])
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
        t_enter = np.maximum.reduce(np.minimum(t1, t2), axis=1)
        t_exit = np.minimum.reduce(np.maximum(t1, t2), axis=1)
        this = (t_exit >= np.maximum(t_enter, eps)) & (t_exit > eps)
        hit |= this
        if hit.all():
            break
    return hit


def _occluded_any(origins, d, boxes, rounded, eps=1e-3):
    """Boolean (N,) — does the ray hit any plain box OR any analytic rounded
    occluder (RoundedRect/RoundedCross; ``rounded`` is empty when
    resist_round<=0, in which case this is exactly `_occluded`)?"""
    hit = _occluded(origins, d, boxes, eps)
    if not rounded:
        return hit
    o = np.asarray(origins, float)
    for ro in rounded:
        if hit.all():
            break
        if isinstance(ro, RoundedCross):
            hit |= _occluded_rounded_cross(o, d, ro, eps)
        else:
            hit |= _occluded_rounded_rect(o, d, ro, eps)
    return hit


def _occluded_mask(ii, jj, kk, d, mask):
    """Boolean (N,): does the ray from each surface voxel (ii, jj, kk) toward the
    source (−d) pass through any True cell of ``mask`` (a prior evaporation's metal,
    i.e. its resist-sidewall coating)?  Marches in ≈1-cell steps along the dominant
    beam axis.  This is what makes a prior deposit narrow the opening seen by a later
    evaporation (the sidewall effect)."""
    Nx, Ny, Nz = mask.shape
    u = -np.asarray(d, float) / (np.abs(d).max() + 1e-12)   # toward source, ~1 cell/step
    fi = ii.astype(float) + 0.5 + u[0]                      # start one step off surface
    fj = jj.astype(float) + 0.5 + u[1]
    fk = kk.astype(float) + 0.5 + u[2]
    hit = np.zeros(len(ii), bool)
    alive = np.ones(len(ii), bool)
    for _ in range(Nx + Ny + Nz):
        ci = np.floor(fi).astype(int); cj = np.floor(fj).astype(int)
        ck = np.floor(fk).astype(int)
        alive &= (ci >= 0) & (ci < Nx) & (cj >= 0) & (cj < Ny) & (ck >= 0) & (ck < Nz)
        if not alive.any():
            break
        a = alive
        hit[a] |= mask[ci[a], cj[a], ck[a]]
        fi += u[0]; fj += u[1]; fk += u[2]
    return hit


# ════════════════════════════════════════════════════════════════
# Voxel simulation
# ════════════════════════════════════════════════════════════════

@dataclass
class DepositionResult:
    xs: np.ndarray            # voxel-centre coords (Nx,)
    ys: np.ndarray
    zs: np.ndarray
    vox: float                # voxel edge [nm]
    solid: np.ndarray         # int8 label grid (Nx,Ny,Nz): EMPTY/RESIST/SUBSTRATE
    al1: np.ndarray           # bool: electrode 1 metal (bilayer: evap1; trilayer: Nb+Al)
    al2: np.ndarray           # bool: electrode 2 metal (bilayer: evap2; trilayer: Al+Nb)
    alox: np.ndarray          # bool: oxide skin on electrode 1
    z_top: float
    meta: dict
    stack: str = "Bilayer"    # "Bilayer" | "Trilayer"
    films: dict = None        # trilayer sub-films: {"nb1","al2","al3","nb4"} → bool grid
    depo_order: np.ndarray = None   # int16 (Nx,Ny,Nz): global deposition step / −1
    depo_frames: list = None        # playback timeline: [{step,label,show_oxide,liftoff}]
    coverage: dict = None           # soft-edge only: {evap label → float32 (Nx,Ny,Nz)},
                                     # continuous pre-quantisation coverage fraction at
                                     # each column's surface-contact voxel, -1 elsewhere
    coverage_sub: dict = None       # soft_supersample>1 only: {evap label → list of
                                     # ns² float32 (Nx,Ny,Nz) grids}, one per lateral
                                     # sub-offset — the un-averaged detail `coverage`
                                     # discards, for a genuinely finer in-plane render

    def idx_x(self, x):  return int(np.clip(np.searchsorted(self.xs, x), 0, len(self.xs) - 1))
    def idx_y(self, y):  return int(np.clip(np.searchsorted(self.ys, y), 0, len(self.ys) - 1))


def _grid_axes(R, z_top, t_metal_tot, max_cells=MAX_CELLS_PER_AXIS, min_vox=6.0):
    span_xy = 2 * R
    z_hi = z_top + t_metal_tot + 60.0
    span_z = z_hi + 40.0
    vox = max(span_xy, span_z) / max_cells
    vox = max(vox, min_vox)                   # finer ray scan ⇒ smaller floor
    xs = np.arange(-R + vox / 2, R, vox)
    ys = np.arange(-R + vox / 2, R, vox)
    zs = np.arange(-vox / 2, z_hi, vox)       # first cell straddles substrate top
    return xs, ys, zs, vox, z_hi


def _label_solid(xs, ys, zs, boxes, rounded=()):
    """Label grid: SUBSTRATE for z<0, RESIST inside any box or analytic
    rounded occluder (RoundedRect/RoundedCross), else EMPTY."""
    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    lab = np.zeros((Nx, Ny, Nz), np.int8)
    lab[:, :, zs < 0] = SUBSTRATE
    X = xs[:, None, None]; Y = ys[None, :, None]; Z = zs[None, None, :]
    for (x0, x1, y0, y1, z0, z1) in boxes:
        inside = ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) &
                  (Z >= z0) & (Z <= z1))
        lab[inside] = RESIST
    for ro in rounded:
        e = _e_of_z(Z, ro.z0, ro.z1, ro.rr, ro.round_top, ro.round_bot)
        in_z = (Z >= ro.z0) & (Z <= ro.z1)
        if isinstance(ro, RoundedCross):
            inA = (np.abs(X) <= ro.arm + e) & (np.abs(Y) <= ro.hy_a + e)
            inB = (np.abs(X) <= ro.hx_b + e) & (np.abs(Y) <= ro.arm + e)
            solid = (~inA) & (~inB) & in_z
        else:
            solid = ~((np.abs(X) <= ro.hx + e) & (np.abs(Y) <= ro.hy + e)) & in_z
        lab[solid] = RESIST
    return lab


def _plassys_dirs(theta, phi, pattern, size, L, K=24, seed=0):
    """Beam-direction cloud for the soft edge from the **real Plassys source**: the
    e-beam raster ``pattern`` of spot ``size`` [mm] on the target at throw distance
    ``L`` [mm].  Each target-plane offset (dx, dy) is mapped to the local incident
    direction by the exact displaced-source geometry (``wafer_local_angles`` with
    ``S0``), identical to the finite-source Monte-Carlo.  Returns ``K`` unit beam
    vectors (all == nominal for a point source / size 0 ⇒ no taper)."""
    offs = sample_beam_cloud(pattern, size, K, np.random.default_rng(seed))
    dirs = []
    for dx, dy in offs:
        lth, lph = wafer_local_angles(theta, phi, 0.0, 0.0, L,
                                      S0=np.array([dx, dy, 0.0]))
        dirs.append(beam_direction(float(lth), float(lph)))
    return dirs


def _cloud_coverage(o, bi, bj, bk, soft, boxes, rounded, occ_mask, sub_offsets, on_pass,
                    return_sub=False):
    """Average lit-fraction over the source-cloud directions ``soft`` for a set
    of surface-cell origins ``o``, optionally supersampled at ``sub_offsets``
    lateral sub-positions within each cell (a smoother in-plane footprint
    boundary — without it every cell is tested from its single centre point,
    which aliases the boundary to the voxel grid).

    Sidewall occlusion (``occ_mask``, via ``_occluded_mask``) is evaluated once
    per ray at the cell's own integer indices, not per sub-offset: it marches
    through ``occ_mask``'s integer voxel grid rather than continuous
    coordinates, so sub-voxel positions don't apply to it the way they do to
    the resist test (``_occluded``, continuous ray–box intersection).

    ``return_sub`` additionally returns the per-sub-offset coverage (averaged
    over the rays in ``soft`` only, NOT over ``sub_offsets``) as an
    ``(len(o), len(sub_offsets))`` array — the un-averaged detail that lets a
    renderer show each lateral sub-position's own value instead of the single
    cell-average, for a genuinely finer in-plane picture near the edge."""
    n_off = len(sub_offsets)
    acc = np.zeros(len(o))
    acc_sub = np.zeros((len(o), n_off)) if return_sub else None
    n_tests = 0
    for dk in soft:
        side_ok = None
        if occ_mask is not None:
            side_ok = ~_occluded_mask(bi, bj, bk, dk, occ_mask)
        for si, off in enumerate(sub_offsets):
            lit_k = ~_occluded_any(o + off, -dk, boxes, rounded)
            if side_ok is not None:
                lit_k &= side_ok
            acc += lit_k.astype(float)
            if return_sub:
                acc_sub[:, si] += lit_k.astype(float)
            n_tests += 1
        if on_pass is not None:
            on_pass()                    # one tick per source ray (not per sub-sample)
    cov = acc / n_tests
    if return_sub:
        return cov, acc_sub / max(1, len(soft))
    return cov


def _deposit(lab, xs, ys, zs, vox, d, t_metal, boxes, rounded=(), occ_mask=None, soft=None,
             record=False, on_pass=None, band_w=3, lateral_supersample=1,
             return_cov=False):
    """Return bool grid of metal voxels for one evaporation.

    A cell receives metal if it is EMPTY, its beam-forward neighbour is solid
    (it sits on a surface facing the beam), and the ray toward the source is
    unobstructed.  The film is then grown `n` cells back toward the source.

    ``occ_mask`` (optional bool grid) is extra occluding metal from a prior
    evaporation: a surface cell is also shadowed if its ray to the source passes
    through it — this is the sidewall effect (prior wall coating narrows the
    opening).  ``None`` ⇒ resist-only shadowing (unchanged behaviour).

    ``soft`` (optional list of unit beam directions = the finite source's angular
    cloud, e.g. from :func:`_plassys_dirs`) turns on the **soft-edge (penumbra)**
    model: occlusion is integrated over those directions, giving a fractional
    coverage per surface cell, and the grown film thickness is tapered to
    ``round(coverage·n)`` — so the penumbra (and any rounded resist lip) yields a
    tapered metal shoulder.  ``None`` ⇒ a single parallel beam (binary, unchanged).

    ``lateral_supersample`` (only relevant when ``soft`` is set): sample an
    ``n×n`` sub-grid of lateral (xy) offsets within each band cell for the
    resist occlusion test instead of just the cell centre, smoothing the
    in-plane footprint boundary at the cost of ``n²`` more ray-box tests.
    ``1`` (default) ⇒ centre-point only, unchanged behaviour.

    ``return_cov`` additionally returns the continuous per-cell coverage
    fraction (before quantisation to an integer layer count) scattered into a
    full-grid float array — see ``simulate()``'s ``coverage`` result field.
    Off by default (no extra cost / memory when not wanted).
    """
    Nx, Ny, Nz = lab.shape
    solid = lab != EMPTY
    sgn = np.sign(d).astype(int)                  # beam-forward sign per axis

    def _fwd_solid(axis, s):
        """Solidity of the forward neighbour (index + s) along one axis."""
        out = np.zeros_like(solid)
        if s > 0:
            sl_o = [slice(None)] * 3; sl_i = [slice(None)] * 3
            sl_o[axis] = slice(0, -1); sl_i[axis] = slice(1, None)
            out[tuple(sl_o)] = solid[tuple(sl_i)]
        elif s < 0:
            sl_o = [slice(None)] * 3; sl_i = [slice(None)] * 3
            sl_o[axis] = slice(1, None); sl_i[axis] = slice(0, -1)
            out[tuple(sl_o)] = solid[tuple(sl_i)]
        return out

    # A cell sits on a deposition surface if the beam, continuing along d, hits
    # solid in ANY forward direction — i.e. the forward neighbour on at least
    # one nonzero beam axis is solid.  Checking each axis (not a single rounded
    # vector) is what lets the tilted beam coat *vertical resist walls*, not
    # just horizontal floors/tops.
    nbr = np.zeros_like(solid)
    for axis in range(3):
        if sgn[axis] != 0:
            nbr |= _fwd_solid(axis, int(sgn[axis]))
    # below the bottom slab the substrate floor counts as solid (downward beam)
    if sgn[2] < 0:
        nbr[:, :, 0] |= (zs[0] < 0)

    # dominant-axis step used only for growing the film back toward the source
    fwd = np.round(d / (np.abs(d).max() + 1e-12)).astype(int)

    surface = (~solid) & nbr
    ii, jj, kk = np.where(surface)
    if len(ii) == 0:
        m0 = np.zeros_like(solid)
        empty_cov = (np.full(solid.shape, -1.0, np.float32), None)
        if record and return_cov:
            return m0, np.full(solid.shape, -1, np.int16), empty_cov
        if record:
            return m0, np.full(solid.shape, -1, np.int16)
        if return_cov:
            return m0, empty_cov
        return m0

    origins = np.stack([xs[ii], ys[jj], zs[kk]], axis=1)
    # Footprint: when soft is None, fixed by the single central beam (unchanged
    # from the parallel-beam model, so the junction area is preserved exactly).
    # When soft is not None, the footprint can also bleed a little past the
    # central beam's edge into the shadow side (see `outer_band` below) — a
    # real extended source partially illuminates just past the nominal shadow
    # line, so the penumbra tapers symmetrically instead of cutting off dead at
    # the central-beam edge.  Either way the soft-edge cone only adjusts film
    # *thickness* near the edge; it never touches the interior.
    N = len(ii)
    # Central beam → the nominal footprint.  Chunk the ray-cast (only when
    # reporting progress) so the bar / ETA advances in fine steps, not one big
    # jump.
    C = 8 if on_pass is not None else 1
    central_lit = np.empty(N, bool)
    for ch in np.array_split(np.arange(N), min(C, N)):
        central_lit[ch] = ~_occluded_any(origins[ch], -d, boxes, rounded)
        if on_pass is not None:
            on_pass()
    if occ_mask is not None:                 # sidewall: prior-metal wall coating shadows
        central_lit &= ~_occluded_mask(ii, jj, kk, d, occ_mask)
    cov = central_lit.astype(float)          # 1.0 lit, 0.0 shadowed
    oidx = np.empty(0, np.int64)
    cov_outer = np.empty(0)
    ns = 1
    want_sub = False
    cov_sub_inner = None
    cov_sub_outer = None
    if soft is not None:
        # Lateral sub-sampling offsets for the resist test (xy only — z stays
        # at the cell centre).  ns=1 ⇒ a single (0,0,0) offset, i.e. today's
        # centre-point-only behaviour, unchanged.
        ns = max(1, int(lateral_supersample))
        if ns > 1:
            off1d = ((np.arange(ns) + 0.5) / ns - 0.5) * vox
            sx, sy = np.meshgrid(off1d, off1d)
            sub_offsets = np.stack([sx.ravel(), sy.ravel(), np.zeros(ns * ns)], axis=1)
        else:
            sub_offsets = np.zeros((1, 3))

        # Penumbra: coverage differs from 1 only NEAR the shadow edge, so cast the
        # K source-cloud rays ONLY on lit cells within `band_w` voxels of a
        # shadowed surface cell.  Interior lit cells stay at coverage 1 (full
        # thickness).  This avoids re-testing every cell with every ray — the main
        # cost of soft-edge — while giving an identical result.
        dark = ~central_lit
        dark3 = np.zeros(lab.shape, bool)
        dark3[ii[dark], jj[dark], kk[dark]] = True
        for _ in range(max(1, int(band_w))):
            dark3 = _dilate6(dark3)
        band = central_lit & dark3[ii, jj, kk]        # lit cells near the edge
        bidx = np.where(band)[0]
        # Per-sub-offset detail (one column of cov per lateral sub-position,
        # not yet averaged together) — only built when both wanted (return_cov)
        # and meaningful (ns>1); interior (non-band) cells default to 1.0,
        # which is exact for every sub-offset there (same reasoning as `cov`'s
        # own band-only restriction above).
        want_sub = return_cov and ns > 1
        cov_sub_inner = np.ones((N, ns * ns), np.float32) if want_sub else None
        if len(bidx):
            bo = origins[bidx]
            bi, bj, bk = ii[bidx], jj[bidx], kk[bidx]
            if want_sub:
                cov_b, cov_sub_b = _cloud_coverage(bo, bi, bj, bk, soft, boxes, rounded,
                                                   occ_mask, sub_offsets, on_pass,
                                                   return_sub=True)
                cov[bidx] = cov_b
                cov_sub_inner[bidx] = cov_sub_b
            else:
                cov[bidx] = _cloud_coverage(bo, bi, bj, bk, soft, boxes, rounded, occ_mask,
                                            sub_offsets, on_pass)
        elif on_pass is not None:
            for _ in soft:
                on_pass()                    # keep the tick count consistent

        # Mirror: shadowed cells within band_w of the LIT region get the same
        # ray-cloud test, so the penumbra bleeds a little past the central-beam
        # edge instead of cutting off dead at it (symmetric, two-sided taper).
        # Inherits one pre-existing, unrelated limitation from the block above
        # unchanged: `surface` (so `ii,jj,kk`) only contains cells whose forward
        # neighbour along the *central* beam's own axis signs is solid, so an
        # off-axis cloud ray could in principle expose a cell outside that set.
        # Not introduced by this change and out of scope here.
        lit3 = np.zeros(lab.shape, bool)
        lit3[ii[central_lit], jj[central_lit], kk[central_lit]] = True
        for _ in range(max(1, int(band_w))):
            lit3 = _dilate6(lit3)
        outer_band = (~central_lit) & lit3[ii, jj, kk]   # shadow-side cells near the edge
        oidx = np.where(outer_band)[0]
        cov_outer = np.zeros(len(oidx))
        cov_sub_outer = np.zeros((len(oidx), ns * ns), np.float32) if want_sub else None
        if len(oidx):
            oo = origins[oidx]
            # Distinct names from the `oi, oj, ok` built later (for the
            # target_outer/growth concatenation) — same cells, different
            # purpose, kept separate so neither shadows the other.
            oi_, oj_, ok_ = ii[oidx], jj[oidx], kk[oidx]
            if want_sub:
                cov_outer, cov_sub_outer = _cloud_coverage(
                    oo, oi_, oj_, ok_, soft, boxes, rounded, occ_mask, sub_offsets, on_pass,
                    return_sub=True)
            else:
                cov_outer = _cloud_coverage(oo, oi_, oj_, ok_, soft, boxes, rounded, occ_mask,
                                            sub_offsets, on_pass)
        elif on_pass is not None:
            for _ in soft:
                on_pass()                    # keep the tick count consistent

    n = int(round(t_metal / vox))
    keep = central_lit                       # nominal lit footprint
    gi, gj, gk = ii[keep], jj[keep], kk[keep]
    # Lit side keeps its floor (>=1 voxel) — unchanged visual behaviour there.
    target_inner = np.maximum(1, np.round(cov[keep] * n)).astype(int)
    # Shadow-side bleed: NO floor — a cell whose cloud coverage rounds to 0
    # layers genuinely gets no metal, which is what lets the taper reach zero a
    # few voxels past the edge instead of guaranteeing a voxel everywhere.
    oi, oj, ok = ii[oidx], jj[oidx], kk[oidx]
    target_outer = np.round(cov_outer * n).astype(int)

    gi = np.concatenate([gi, oi]); gj = np.concatenate([gj, oj]); gk = np.concatenate([gk, ok])
    target = np.concatenate([target_inner, target_outer])
    cov_all = np.concatenate([cov[keep], cov_outer])   # parallel to gi/gj/gk/target
    if want_sub:
        cov_sub_all = np.concatenate([cov_sub_inner[keep], cov_sub_outer], axis=0)

    metal = np.zeros_like(solid)
    # step-0 (surface-contact) placement is gated on target>=1: inner cells are
    # always >=1 (floored above) so this is a no-op for them there; outer cells
    # with 0 coverage correctly get no metal at all.
    place = target >= 1
    gi_p, gj_p, gk_p = gi[place], gj[place], gk[place]
    metal[gi_p, gj_p, gk_p] = True
    order = np.full(solid.shape, -1, np.int16) if record else None
    if record:
        order[gi_p, gj_p, gk_p] = 0          # surface-contact layer = step 0
    cov_grid = None
    cov_sub_grids = None
    if return_cov:
        # Continuous coverage (pre-quantisation), one value per column at its
        # surface-contact (step-0) voxel — the same fraction `target` above was
        # rounded from, kept at full precision instead of discarded.
        cov_grid = np.full(solid.shape, -1.0, np.float32)
        cov_grid[gi_p, gj_p, gk_p] = cov_all[place]
        if want_sub:
            # One grid per lateral sub-offset (ns² of them) — the un-averaged
            # detail a renderer can use to show genuinely finer in-plane
            # structure near the edge instead of one value per coarse voxel.
            cov_sub_grids = []
            for sidx in range(ns * ns):
                g = np.full(solid.shape, -1.0, np.float32)
                g[gi_p, gj_p, gk_p] = cov_sub_all[place, sidx]
                cov_sub_grids.append(g)

    # grow film thickness back toward source; the soft edge tapers each column to
    # round(coverage·n) so the penumbra reads as a rounded metal shoulder.
    cur_i, cur_j, cur_k = gi.copy(), gj.copy(), gk.copy()
    for mstep in range(1, max(n, 1)):
        cur_i = cur_i - fwd[0]; cur_j = cur_j - fwd[1]; cur_k = cur_k - fwd[2]
        grow = ((target > mstep) &
                (cur_i >= 0) & (cur_i < Nx) & (cur_j >= 0) & (cur_j < Ny) &
                (cur_k >= 0) & (cur_k < Nz))
        ci, cj, ck = cur_i[grow], cur_j[grow], cur_k[grow]
        free = ~solid[ci, cj, ck]
        metal[ci[free], cj[free], ck[free]] = True
        if record:
            order[ci[free], cj[free], ck[free]] = mstep

    if record and return_cov:
        return metal, order, (cov_grid, cov_sub_grids)
    if record:
        return metal, order
    if return_cov:
        return metal, (cov_grid, cov_sub_grids)
    return metal


def simulate(p: ProcessParams, max_cells: int = MAX_CELLS_PER_AXIS,
             min_vox: float = 6.0, record: bool = False,
             progress=None) -> DepositionResult:
    """Run the shadow-evaporation engine.

    ``max_cells`` / ``min_vox`` set the ray-scan resolution: a larger
    ``max_cells`` (and smaller ``min_vox`` floor) traces the beam into a finer
    voxel grid — more accurate metal/junction edges at the cost of speed/memory.

    ``record`` (opt-in) additionally captures a per-voxel **deposition timeline**
    (``depo_order`` + ``depo_frames``) for the step-through playback — each film's
    growth layer is tagged with a global step, so a frame at step k shows the metal
    deposited so far.  Off by default (no extra cost on the wafer/scan/MC paths).
    """
    boxes, R, z_top, resist_h, z_split, rounded = build_occluders(p)
    trilayer = getattr(p, "stack", "Bilayer") == "Trilayer"
    if trilayer:
        metal_sum = p.tri_t1 + p.tri_t2 + p.tri_t3 + p.tri_t4
        t_tot_metal = metal_sum + max(p.tri_t1 * 0.1, 3)
    else:
        metal_sum = p.t_metal1 + p.t_metal2
        t_tot_metal = metal_sum + max(p.t_metal1 * 0.1, 3)

    # The voxel grid only needs to resolve the region we actually observe
    # (the junction and the nearby leads).  Occluder boxes are analytic AABBs
    # that always extend to the full opening (±R), so beams are still admitted
    # or blocked correctly even when the grid is cropped to a smaller window.
    # Cropping lets us spend resolution where it matters: for Manhattan the
    # opening must be very long so a 60° beam can reach the floor, which would
    # otherwise force ~100 nm voxels — coarser than the 30 nm film, leaving
    # patchy / missing metal.
    if p.mode == "Dolan bridge":
        grid_R = R
    else:
        span = max(p.manhattan_wx, p.manhattan_wy)
        grid_R = min(R, 1.6 * span + 800.0)
    xs, ys, zs, vox, z_hi = _grid_axes(grid_R, z_top, t_tot_metal,
                                       max_cells=max_cells, min_vox=min_vox)
    lab = _label_solid(xs, ys, zs, boxes, rounded)

    def _oxide_skin(metal):
        """One-cell conformal oxide skin on all exposed faces of `metal`."""
        neigh = np.zeros_like(metal)
        neigh[1:, :, :] |= metal[:-1, :, :]; neigh[:-1, :, :] |= metal[1:, :, :]
        neigh[:, 1:, :] |= metal[:, :-1, :]; neigh[:, :-1, :] |= metal[:, 1:, :]
        neigh[:, :, 1:] |= metal[:, :, :-1]; neigh[:, :, :-1] |= metal[:, :, 1:]
        return neigh & (~metal) & (lab == EMPTY)

    # Both modes drive the evaporations from their own (θ, φ).  Dolan
    # defaults to a uniaxial ±θ tilt at φ=0; Manhattan defaults to orthogonal
    # azimuths (φ₁=0, φ₂=90).  All angles are free.
    films = None
    tri_dirs = None
    # Soft-edge (penumbra): per-evaporation beam-direction cloud from the real
    # Plassys source (e-beam pattern at the throw distance), or None for a single
    # beam.  Built per evap from its own nominal (θ, φ).
    _soft_on = getattr(p, "soft_edge", False) and p.soft_size > 0 and p.soft_L > 0

    def _soft_cloud(theta, phi):
        if not _soft_on:
            return None
        return _plassys_dirs(theta, phi, p.soft_pattern, p.soft_size, p.soft_L,
                             K=p.soft_rays)

    # Live progress: per evaporation the central beam is cast in C=8 chunks
    # (fine-grained ticks) plus one tick per soft-cloud ray → finer ETA than one
    # tick per evap.  `progress=None` ⇒ no overhead.  Penumbra band half-width
    # `_band_w` (voxels) ≈ source half-angle × lip height, used to restrict the
    # soft rays to the shadow-edge band.
    _C = 8
    _n_evap = 4 if trilayer else 2
    if _soft_on:
        _cloud0 = _plassys_dirs(p.angle1, p.phi1, p.soft_pattern, p.soft_size, p.soft_L,
                                K=p.soft_rays)
        _rays = len(_cloud0)
        _d0 = beam_direction(p.angle1, p.phi1)       # max angular deviation of the
        _dev = np.arccos(np.clip([float(np.dot(dk, _d0)) for dk in _cloud0],
                                 -1.0, 1.0)).max()    # actual cloud (covers tails)
        _band_w = int(np.ceil(np.tan(float(_dev)) * z_top / vox)) + 2
    else:
        _rays, _band_w = 0, 3
    _U = max(1, _n_evap * (_C + 2 * _rays))   # ×2: inner + mirrored outer-band ray loop
    _done = [0]

    def _on_pass_for(label):
        if progress is None:
            return None
        def _op():
            _done[0] += 1
            progress(min(_done[0] / _U, 1.0), label)
        return _op

    # Playback recording: each evaporation appends (order_grid, label, thickness_nm)
    # in deposition order; `_ox_after` = index of the evap after which oxidation
    # happens (oxide appears in frames past it).
    _rec = []
    _cov_rec = {}                            # label -> continuous coverage grid
    _cov_sub_rec = {}                         # label -> list of ns² per-sub-offset grids
    _lat_sub = getattr(p, "soft_supersample", 1)

    def _dep(lab_, d_, t_, occ_, soft_, label):
        op = _on_pass_for(label)
        if record:
            m, o, (cg, csub) = _deposit(lab_, xs, ys, zs, vox, d_, t_, boxes, rounded,
                                        occ_mask=occ_, soft=soft_, record=True, return_cov=True,
                                        on_pass=op, band_w=_band_w, lateral_supersample=_lat_sub)
            _rec.append((o, label, float(t_)))
            if soft_ is not None:
                _cov_rec[label] = cg
                if csub is not None:
                    _cov_sub_rec[label] = csub
            return m
        return _deposit(lab_, xs, ys, zs, vox, d_, t_, boxes, rounded,
                        occ_mask=occ_, soft=soft_, on_pass=op, band_w=_band_w,
                        lateral_supersample=_lat_sub)

    if not trilayer:
        # ── Bilayer: evap1 → oxidation → evap2 ────────────────────
        d1 = beam_direction(p.angle1, p.phi1)
        d2 = beam_direction(p.angle2, p.phi2)
        al1 = _dep(lab, d1, p.t_metal1, None, _soft_cloud(p.angle1, p.phi1),
                   "Evap 1")
        # Oxide: thin conformal skin on al1; NOT geometry (does not occlude).
        alox = _oxide_skin(al1)
        lab2 = lab.copy()
        lab2[al1] = RESIST
        al2 = _dep(lab2, d2, p.t_metal2, (al1 if p.sidewall else None),
                   _soft_cloud(p.angle2, p.phi2), "Evap 2")
        meta_d1, meta_d2 = d1, d2
        _ox_after = 0                                  # oxidation after evap 1
    else:
        # ── Trilayer: Nb1 → Al2 → oxidation → Al3 → Nb4 ───────────
        # Electrode 1 (bottom) = Nb(evap1) + Al(evap2) at the Evap-1 angle;
        # electrode 2 (top)    = Al(evap3) + Nb(evap4) at the Evap-2 angle.
        d1 = beam_direction(p.angle1, p.phi1)        # evap1  Nb
        d2 = beam_direction(p.tri_angle2, p.tri_phi2)  # evap2  Al
        d3 = beam_direction(p.angle2, p.phi2)        # evap3  Al
        d4 = beam_direction(p.tri_angle4, p.tri_phi4)  # evap4  Nb

        nb1 = _dep(lab, d1, p.tri_t1, None, _soft_cloud(p.angle1, p.phi1),
                   "Evap 1 — Nb")
        lab_b = lab.copy(); lab_b[nb1] = RESIST
        al2f = _dep(lab_b, d2, p.tri_t2, (nb1 if p.sidewall else None),
                    _soft_cloud(p.tri_angle2, p.tri_phi2), "Evap 2 — Al")
        elec1 = nb1 | al2f                            # bottom electrode

        # Oxidation after evap1+evap2: skin on ALL exposed faces of the bottom
        # electrode — i.e. on both the Nb and the Al that face free space.
        alox = _oxide_skin(elec1)

        lab_c = lab.copy(); lab_c[elec1] = RESIST     # electrode 2 sees elec1 solid
        al3 = _dep(lab_c, d3, p.tri_t3, (elec1 if p.sidewall else None),
                   _soft_cloud(p.angle2, p.phi2), "Evap 3 — Al")
        lab_d = lab_c.copy(); lab_d[al3] = RESIST
        nb4 = _dep(lab_d, d4, p.tri_t4, ((elec1 | al3) if p.sidewall else None),
                   _soft_cloud(p.tri_angle4, p.tri_phi4), "Evap 4 — Nb")
        elec2 = al3 | nb4                             # top electrode
        _ox_after = 1                                  # oxidation after evap 2 (Al)

        al1, al2 = elec1, elec2                       # keep al1/al2 = the two electrodes
        films = dict(nb1=nb1, al2=al2f, al3=al3, nb4=nb4)
        meta_d1, meta_d2 = d1, d3                     # arrows: the two electrode beams
        tri_dirs = dict(nb1=d1, al2=d2, al3=d3, nb4=d4)  # per-evaporation beams

    if progress is not None:
        progress(1.0, "Deposits complete")   # the lift-off connectivity flood fill
                                              # (a second, separately-timed phase) is
                                              # reported by the caller around
                                              # vv._grounded_metal, not from here

    z_floor = metal_sum + max(p.tri_t1 * 0.1 if trilayer else p.t_metal1 * 0.1, 3) + 2 * vox
    # Junction region: where the device tunnel junction is expected.  In Dolan
    # the open trench floods both depositions onto the leads; the actual JJ is
    # the under-bridge overlap, so confine the measurement to the bridge zone.
    if p.mode == "Dolan bridge":
        reach = p.t_mma * np.tan(np.radians(max(abs(p.angle1), abs(p.angle2))))
        junc_xmax = p.bridge_len / 2 + reach + 2 * vox
        junc_ymax = p.bridge_w / 2 + 2 * vox
    else:
        junc_xmax = p.manhattan_wy / 2 + 4 * vox
        junc_ymax = p.manhattan_wx / 2 + 4 * vox
    z_floor_v = z_floor
    meta = dict(R=R, grid_R=grid_R, z_top=z_top, resist_h=resist_h, vox=vox,
                d1=meta_d1, d2=meta_d2, z_floor=z_floor_v, z_split=z_split,
                junc_xmax=junc_xmax, junc_ymax=junc_ymax, mode=p.mode,
                stack="Trilayer" if trilayer else "Bilayer",
                tri_dirs=tri_dirs,
                max_cells=max_cells, min_vox=min_vox,
                soft_supersample=_lat_sub,
                resist_round=getattr(p, "resist_round", 0.0), rounded_geom=rounded)

    # ── Playback timeline (when recording) ────────────────────────
    depo_order = depo_frames = None
    if record:
        depo_order = np.full(lab.shape, -1, np.int16)
        depo_frames = [dict(step=-1, label="Resist (before evaporation)",
                            show_oxide=False, liftoff=False)]
        g = 0
        for idx, (o, label, t_nm) in enumerate(_rec):
            m = o >= 0
            steps = int(o[m].max()) + 1 if m.any() else 1
            depo_order[m] = g + o[m].astype(np.int16)   # global step of each voxel
            for s in range(steps):
                thick = min((s + 1) * vox, t_nm)
                depo_frames.append(dict(
                    step=g + s, label=f"{label} — {thick:.0f}/{t_nm:.0f} nm",
                    show_oxide=(idx > _ox_after), liftoff=False))
            g += steps
            if idx == _ox_after:
                depo_frames.append(dict(step=g - 1, label="Oxidation",
                                        show_oxide=True, liftoff=False))
        depo_frames.append(dict(step=g, label="Lift-off (resist stripped)",
                                show_oxide=True, liftoff=True))

    return DepositionResult(xs, ys, zs, vox, lab, al1, al2, alox, z_top, meta,
                            stack="Trilayer" if trilayer else "Bilayer",
                            films=films, depo_order=depo_order,
                            depo_frames=depo_frames,
                            coverage=(_cov_rec or None) if record else None,
                            coverage_sub=(_cov_sub_rec or None) if record else None)


# ════════════════════════════════════════════════════════════════
# Measurements from the voxel result
# ════════════════════════════════════════════════════════════════

def _connected_components(mask):
    """4-connectivity labelling of a 2-D boolean mask (pure NumPy, no scipy).

    Returns (labels, n) with labels.shape == mask.shape, 0 = background and
    1..n the component ids.  Uses union-find over the True cells.
    """
    n0, n1 = mask.shape
    a0, a1 = np.where(mask)
    m = len(a0)
    if m == 0:
        return np.zeros(mask.shape, np.int64), 0
    idx = np.full(mask.shape, -1, np.int64)
    idx[a0, a1] = np.arange(m)
    parent = np.arange(m)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:        # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for c in range(m):
        i, j = a0[c], a1[c]
        if i + 1 < n0 and mask[i + 1, j]:
            union(c, int(idx[i + 1, j]))
        if j + 1 < n1 and mask[i, j + 1]:
            union(c, int(idx[i, j + 1]))

    roots = np.array([find(c) for c in range(m)])
    uniq, inv = np.unique(roots, return_inverse=True)
    labels = np.zeros(mask.shape, np.int64)
    labels[a0, a1] = inv + 1
    return labels, len(uniq)


def _dilate6(m):
    """6-neighbour (face) dilation of a 3-D boolean mask."""
    out = np.zeros_like(m)
    out[1:, :, :] |= m[:-1, :, :]; out[:-1, :, :] |= m[1:, :, :]
    out[:, 1:, :] |= m[:, :-1, :]; out[:, :-1, :] |= m[:, 1:, :]
    out[:, :, 1:] |= m[:, :, :-1]; out[:, :, :-1] |= m[:, :, 1:]
    return out


def _junction_cells_3d(r: DepositionResult, min_cells: int = 2,
                       include_sidewalls: bool = True):
    """Cleaned 3-D Josephson-junction mask + per-junction list.

    The tunnel barrier is the thin oxide between the two electrodes.  In the
    voxel model that is every oxide-skin cell of electrode 1 that electrode-2
    metal actually reaches (``alox & al2``) — which happens on the substrate
    FLOOR *and* on the vertical metal SIDEWALLS wherever the second beam coats
    them.  Restricted to the in-trench metal stack (0 ≤ z < z_floor) so stray
    metal on top of the resist is not mistaken for a junction.

    Separate *devices* are grouped by xy-footprint connectivity — a junction and
    its sidewalls share one footprint (a wall sits one cell outside the floor
    edge, so it is 4-connected to it), so the floor + its walls count as ONE
    junction — while the area is measured in full 3-D (floor + walls).

    Returns ``(clean3d, juncs)``: ``clean3d`` the 3-D boolean junction mask, and
    ``juncs`` a list (largest-area first) of per-junction dicts with keys
    ``mask`` (xy footprint), ``mask3d``, ``cells``, ``area`` (= cells·vox²,
    INCLUDING walls), ``ox``, ``oy``, ``cx``, ``cy``.
    """
    zs = r.zs
    z_floor = r.meta.get("z_floor", r.z_top)
    zsel = (zs >= 0) & (zs < z_floor)
    j3 = r.alox & r.al2 & zsel[None, None, :]    # oxide cells reached by elec 2
    if not include_sidewalls:
        # Floor-only: keep only the horizontal top-surface barrier — oxide cells
        # with bottom-electrode metal directly beneath (elec 2 on top of elec 1).
        # Vertical sidewall barriers (oxide on a wall face, metal/resist below,
        # not elec-1 metal) are dropped.
        al1_below = np.zeros_like(r.al1)
        al1_below[:, :, 1:] = r.al1[:, :, :-1]
        j3 = j3 & al1_below
    j2 = j3.any(axis=2)                           # xy footprint of the junction

    # Dolan's open trench floods both depositions onto the leads, leaving stray
    # pad overlaps far out in x; keep only blobs whose centroid sits in the
    # bridge junction zone.  Manhattan crosses exactly once → keep every blob.
    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    confine = r.meta.get("mode", "") != "Manhattan"

    labels, n = _connected_components(j2)         # device-level (xy) grouping
    juncs = []
    clean = np.zeros_like(j3)
    for lab_id in range(1, n + 1):
        comp2 = labels == lab_id                  # xy footprint of this junction
        ci, cj = np.where(comp2)
        cx = float(r.xs[ci].mean())
        cy = float(r.ys[cj].mean())
        if confine and (abs(cx) > jxm or abs(cy) > jym):   # stray pad overlap
            continue
        cells3d = j3 & comp2[:, :, None]          # floor + walls of this device
        cnt = int(cells3d.sum())
        if cnt < min_cells:
            continue
        clean |= cells3d
        ox_c = (ci.max() - ci.min() + 1) * r.vox
        oy_c = (cj.max() - cj.min() + 1) * r.vox
        juncs.append(dict(
            mask=comp2, mask3d=cells3d, cells=cnt, area=cnt * r.vox * r.vox,
            ox=ox_c, oy=oy_c, cx=cx, cy=cy,
        ))
    juncs.sort(key=lambda d: -d["area"])
    return clean, juncs


def junction_footprint(r: DepositionResult, min_cells: int = 2,
                       include_sidewalls: bool = True):
    """Full 3-D Josephson-junction barrier: substrate floor *and* metal walls.

    The barrier is every oxide cell electrode 2 reaches across the oxide skin of
    electrode 1 — on the floor and on the vertical sidewalls of the lower
    electrode (see ``_junction_cells_3d``).  A device can contain several
    spatially separate junctions (e.g. one on each side of a Dolan bridge for
    some tilt angles); each is its own entry in ``juncs`` (largest-area first).

    Returns ``(mask_xy, area_nm2, ox_nm, oy_nm, juncs)``.  ``mask_xy`` is the xy
    projection of the 3-D junction (for the top-view map); ``area_nm2`` is the
    full 3-D barrier area (floor + walls); ``ox/oy`` are the xy footprint of the
    largest junction (not a bounding box merging separate junctions).
    """
    clean, juncs = _junction_cells_3d(r, min_cells, include_sidewalls)
    junc = clean.any(axis=2)                       # xy projection for the map
    area = sum(d["area"] for d in juncs)
    if juncs:
        ox, oy = juncs[0]["ox"], juncs[0]["oy"]
    else:
        ox = oy = 0.0
    return junc, area, ox, oy, juncs


# combination codes for the trilayer junction map
COMBO_NONE, COMBO_NBAL, COMBO_ALAL, COMBO_NBNB = 0, 1, 2, 3
_COMBO_NAMES = {COMBO_NBAL: "Nb-Al", COMBO_ALAL: "Al-Al", COMBO_NBNB: "Nb-Nb"}


def junction_combos(r: DepositionResult, min_cells: int = 2,
                    include_sidewalls: bool = True):
    """Classify every 3-D junction cell by the metal pair across the oxide.

    For a trilayer (Nb/Al–AlOx–Al/Nb) each barrier cell separates electrode-1's
    metal (Al where the evap-2 Al sublayer borders the cell, else the evap-1 Nb)
    from electrode-2's metal (Al where the evap-3 Al fills the cell, else the
    evap-4 Nb).  The pair is binned (order-independent) into Nb-Al / Al-Al /
    Nb-Nb.  Classification runs over the SAME 3-D junction cells measured by
    ``junction_footprint`` (floor + walls), so the per-pair areas sum to the
    headline area.

    Returns ``(combos, combo_map)`` where ``combos`` maps name → {mask, cells,
    area} (area = 3-D, incl. walls) and ``combo_map`` is an (Nx,Ny) int grid of
    COMBO_* codes (per-column dominant pair, for the top-view map).  For a
    bilayer result returns ``({}, None)``.
    """
    if getattr(r, "stack", "Bilayer") != "Trilayer" or not r.films:
        return {}, None
    clean, _ = _junction_cells_3d(r, min_cells, include_sidewalls)

    # electrode-1 side: Al where the evap-2 Al sublayer borders the oxide cell,
    # else Nb (every barrier cell is an oxide-skin cell of electrode 1, so the
    # non-Al cells necessarily border the evap-1 Nb).
    e1_al = clean & _dilate6(r.films["al2"])
    # electrode-2 side: Al where the evap-3 Al fills the cell, else evap-4 Nb.
    e2_al = clean & r.films["al3"]

    alal = e1_al & e2_al
    nbnb = clean & (~e1_al) & (~e2_al)
    nbal = clean & (~alal) & (~nbnb)            # one Al + one Nb

    combo3d = np.zeros(clean.shape, np.int8)
    combo3d[nbal] = COMBO_NBAL
    combo3d[alal] = COMBO_ALAL
    combo3d[nbnb] = COMBO_NBNB

    # 2-D map for the top view: per xy column take the most-represented pair.
    counts = np.stack([(combo3d == COMBO_NBAL).sum(axis=2),
                       (combo3d == COMBO_ALAL).sum(axis=2),
                       (combo3d == COMBO_NBNB).sum(axis=2)], axis=-1)
    codes = np.array([COMBO_NBAL, COMBO_ALAL, COMBO_NBNB], np.int8)
    combo_map = np.zeros(clean.shape[:2], np.int8)
    has = counts.sum(axis=-1) > 0
    combo_map[has] = codes[counts.argmax(axis=-1)[has]]

    combos = {}
    for code, name, m3 in ((COMBO_NBAL, "Nb-Al", nbal),
                           (COMBO_ALAL, "Al-Al", alal),
                           (COMBO_NBNB, "Nb-Nb", nbnb)):
        cnt = int(m3.sum())
        if cnt < min_cells:
            continue
        combos[name] = dict(mask=(combo_map == code), cells=cnt,
                            area=cnt * r.vox * r.vox)
    return combos, combo_map
