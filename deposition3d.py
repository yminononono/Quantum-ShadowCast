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

from process_engine import ProcessParams


# ── voxel labels ────────────────────────────────────────────────
EMPTY = 0
RESIST = 1
SUBSTRATE = 2

MAX_CELLS_PER_AXIS = 110   # resolution cap (keeps the grid tractable)


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


def build_occluders(p: ProcessParams):
    """Return (boxes, R, z_top, resist_h) for the given process."""
    if p.mode == "Dolan bridge":
        L = p.bridge_len                            # suspended bridge width (x)
        Wj = p.bridge_w                             # junction width (y)
        u = p.undercut
        t_mma, t_pmma = p.t_mma, p.t_pmma
        z_top = t_mma + t_pmma

        # Open windows must sit on either side of the bridge so the tilted
        # beam can reach the floor and slide under the bridge.  Make the trench
        # comfortably wider than the under-bridge reach (t_mma·tanθ).
        # Window must be wide enough that the far electrode wall does not
        # shadow the under-bridge region; then the junction is bridge-limited
        # (the intended Dolan behaviour, overlap ≈ 2·t_mma·tanθ − L).
        tanmax = np.tan(np.radians(max(abs(p.angle1), abs(p.angle2))))
        window = tanmax * z_top + L + 250.0
        trench_hx = L / 2 + window                  # PMMA trench half-width (x)
        ap_hy = Wj / 2                              # aperture half-width (y)
        R = trench_hx + u + 400.0

        boxes = []
        # MMA layer: trench widened by undercut on all sides
        boxes += _frame_boxes(trench_hx + u, ap_hy + u, 0.0, t_mma, R)
        # PMMA layer: open trench (no bridge yet)
        boxes += _frame_boxes(trench_hx, ap_hy, t_mma, z_top, R)
        # Suspended bridge: narrow strip across the middle of the trench,
        # hanging over the MMA air gap.
        boxes.append(_box(-L / 2, L / 2, -ap_hy, ap_hy, t_mma, z_top))
        return boxes, R, z_top, t_mma, t_mma   # z_split = MMA/PMMA interface

    else:  # Manhattan double-oblique: two perpendicular resist lines crossing
        wA = p.manhattan_wx        # electrode A linewidth (A runs along x)
        wB = p.manhattan_wy        # electrode B linewidth (B runs along y)
        h = p.manhattan_h          # upper imaging-resist thickness
        u = p.undercut             # one-sided lateral undercut
        # Bilayer: a lower undercut sublayer (widened openings) carries the upper
        # imaging resist (nominal openings).  The undercut shelf lets metal lift
        # off cleanly, exactly like the Dolan MMA/PMMA stack.
        t_lo = min(0.3 * h, 400.0)                   # undercut sublayer thickness
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
        boxes = _solid_from_openings([A_u, B_u], R, 0.0, t_lo)      # lower (undercut)
        boxes += _solid_from_openings([A, B], R, t_lo, z_top)       # upper (imaging)
        return boxes, R, z_top, z_top, t_lo    # z_split = undercut/imaging interface


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
    al1: np.ndarray           # bool: Al from evaporation 1
    al2: np.ndarray           # bool: Al from evaporation 2
    alox: np.ndarray          # bool: AlOx layer
    z_top: float
    meta: dict

    def idx_x(self, x):  return int(np.clip(np.searchsorted(self.xs, x), 0, len(self.xs) - 1))
    def idx_y(self, y):  return int(np.clip(np.searchsorted(self.ys, y), 0, len(self.ys) - 1))


def _grid_axes(R, z_top, t_metal_tot):
    span_xy = 2 * R
    z_hi = z_top + t_metal_tot + 60.0
    span_z = z_hi + 40.0
    vox = max(span_xy, span_z) / MAX_CELLS_PER_AXIS
    vox = max(vox, 6.0)                       # don't go finer than 6 nm
    xs = np.arange(-R + vox / 2, R, vox)
    ys = np.arange(-R + vox / 2, R, vox)
    zs = np.arange(-vox / 2, z_hi, vox)       # first cell straddles substrate top
    return xs, ys, zs, vox, z_hi


def _label_solid(xs, ys, zs, boxes):
    """Label grid: SUBSTRATE for z<0, RESIST inside any box, else EMPTY."""
    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    lab = np.zeros((Nx, Ny, Nz), np.int8)
    lab[:, :, zs < 0] = SUBSTRATE
    X = xs[:, None, None]; Y = ys[None, :, None]; Z = zs[None, None, :]
    for (x0, x1, y0, y1, z0, z1) in boxes:
        inside = ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) &
                  (Z >= z0) & (Z <= z1))
        lab[inside] = RESIST
    return lab


def _deposit(lab, xs, ys, zs, vox, d, t_metal, boxes):
    """Return bool grid of metal voxels for one evaporation.

    A cell receives metal if it is EMPTY, its beam-forward neighbour is solid
    (it sits on a surface facing the beam), and the ray toward the source is
    unobstructed.  The film is then grown `n` cells back toward the source.
    """
    Nx, Ny, Nz = lab.shape
    solid = lab != EMPTY

    # beam-forward neighbour index offset (one voxel step along +d)
    step = np.sign(d) * (np.abs(d) > 0.3)        # dominant-axis stepping
    # use a finer forward probe: shift by one cell along each nonzero axis
    fwd = np.round(d / (np.abs(d).max() + 1e-12)).astype(int)

    occ = np.zeros_like(solid)
    sx, sy, sz = fwd
    # shifted solidity: neighbour = cell at (i+sx, j+sy, k+sz)
    nbr = np.zeros_like(solid)
    xi = slice(max(0, -sx), Nx - max(0, sx))
    xo = slice(max(0, sx), Nx - max(0, -sx))
    yi = slice(max(0, -sy), Ny - max(0, sy))
    yo = slice(max(0, sy), Ny - max(0, -sy))
    zi = slice(max(0, -sz), Nz - max(0, sz))
    zo = slice(max(0, sz), Nz - max(0, -sz))
    nbr[xi, yi, zi] = solid[xo, yo, zo]
    # forward neighbour out of domain on the -z side counts as substrate floor
    if sz < 0:
        nbr[:, :, :(-sz)] |= (zs[:(-sz)] < 0)[None, None, :]

    surface = (~solid) & nbr
    ii, jj, kk = np.where(surface)
    if len(ii) == 0:
        return np.zeros_like(solid)

    origins = np.stack([xs[ii], ys[jj], zs[kk]], axis=1)
    u = -d                                   # toward the source
    blocked = _occluded(origins, u, boxes)
    lit = ~blocked

    metal = np.zeros_like(solid)
    gi, gj, gk = ii[lit], jj[lit], kk[lit]
    metal[gi, gj, gk] = True

    # grow film thickness back toward source
    n = int(round(t_metal / vox))
    cur_i, cur_j, cur_k = gi.copy(), gj.copy(), gk.copy()
    for _ in range(max(n - 1, 0)):
        cur_i = cur_i - fwd[0]; cur_j = cur_j - fwd[1]; cur_k = cur_k - fwd[2]
        m = ((cur_i >= 0) & (cur_i < Nx) & (cur_j >= 0) & (cur_j < Ny) &
             (cur_k >= 0) & (cur_k < Nz))
        ci, cj, ck = cur_i[m], cur_j[m], cur_k[m]
        free = ~solid[ci, cj, ck]
        metal[ci[free], cj[free], ck[free]] = True
    return metal


def simulate(p: ProcessParams) -> DepositionResult:
    boxes, R, z_top, resist_h, z_split = build_occluders(p)
    t_tot_metal = p.t_metal1 + p.t_metal2 + max(p.t_metal1 * 0.1, 3)
    xs, ys, zs, vox, z_hi = _grid_axes(R, z_top, t_tot_metal)
    lab = _label_solid(xs, ys, zs, boxes)

    # Both modes drive the two evaporations from their own (θ, φ).  Dolan
    # defaults to a uniaxial ±θ tilt at φ=0; Manhattan defaults to two beams at
    # the same tilt but orthogonal azimuth (φ₁=0, φ₂=90).  All four are free.
    d1 = beam_direction(p.angle1, p.phi1)
    d2 = beam_direction(p.angle2, p.phi2)

    al1 = _deposit(lab, xs, ys, zs, vox, d1, p.t_metal1, boxes)

    # oxide: a thin skin on top of al1 (one cell toward source of beam 1)
    alox = np.zeros_like(al1)
    fwd1 = np.round(d1 / (np.abs(d1).max() + 1e-12)).astype(int)
    ai, aj, ak = np.where(al1)
    oi, oj, ok = ai - fwd1[0], aj - fwd1[1], ak - fwd1[2]
    m = ((oi >= 0) & (oi < lab.shape[0]) & (oj >= 0) & (oj < lab.shape[1]) &
         (ok >= 0) & (ok < lab.shape[2]))
    oi, oj, ok = oi[m], oj[m], ok[m]
    free = (lab[oi, oj, ok] == EMPTY) & (~al1[oi, oj, ok])
    alox[oi[free], oj[free], ok[free]] = True

    # evaporation 2 sees al1+alox as additional solid surface
    lab2 = lab.copy()
    lab2[al1] = RESIST
    lab2[alox] = RESIST
    al2 = _deposit(lab2, xs, ys, zs, vox, d2, p.t_metal2, boxes)

    z_floor = p.t_metal1 + p.t_metal2 + max(p.t_metal1 * 0.1, 3) + 2 * vox
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
    meta = dict(R=R, z_top=z_top, resist_h=resist_h, vox=vox,
                d1=d1, d2=d2, z_floor=z_floor_v, z_split=z_split,
                junc_xmax=junc_xmax, junc_ymax=junc_ymax)
    return DepositionResult(xs, ys, zs, vox, lab, al1, al2, alox, z_top, meta)


# ════════════════════════════════════════════════════════════════
# Measurements from the voxel result
# ════════════════════════════════════════════════════════════════

def junction_footprint(r: DepositionResult):
    """xy cells (on the substrate floor) where Al1, AlOx and Al2 stack up.

    Returns (mask_xy, area_nm2, ox_nm, oy_nm) where the junction is the set of
    columns in which an Al1 voxel, an AlOx voxel above it, and an Al2 voxel
    above that all occur near the floor (under-bridge / line-crossing region).
    """
    zs = r.zs
    z_floor = r.meta.get("z_floor", r.z_top)
    floor = (zs >= 0) & (zs < z_floor)           # metal sitting on the substrate
    al1f = r.al1[:, :, floor].any(axis=2)
    al2f = r.al2[:, :, floor].any(axis=2)
    junc = al1f & al2f
    # confine to the device junction region
    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    reg = (np.abs(r.xs)[:, None] <= jxm) & (np.abs(r.ys)[None, :] <= jym)
    junc = junc & reg
    area = junc.sum() * r.vox * r.vox
    # extents
    if junc.any():
        xi = np.where(junc.any(axis=1))[0]
        yi = np.where(junc.any(axis=0))[0]
        ox = (xi.max() - xi.min() + 1) * r.vox
        oy = (yi.max() - yi.min() + 1) * r.vox
    else:
        ox = oy = 0.0
    return junc, area, ox, oy
