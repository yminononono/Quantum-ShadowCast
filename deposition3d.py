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


def build_occluders(p: ProcessParams):
    """Return (boxes, R, z_top, resist_h) for the given process."""
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

        boxes = []
        # MMA (undercut sublayer) fills 0..gap; trench widened by undercut.
        boxes += _frame_boxes(trench_hx + u, ap_hy + u, 0.0, gap, R)
        # PMMA layer: open trench (no bridge yet), gap..z_top
        boxes += _frame_boxes(trench_hx, ap_hy, gap, z_top, R)
        # Suspended bridge: narrow strip across the middle of the trench,
        # hanging over the air gap at z = gap.
        boxes.append(_box(-L / 2, L / 2, -ap_hy, ap_hy, gap, z_top))
        return boxes, R, z_top, gap, gap   # z_split = MMA/PMMA interface

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


def simulate(p: ProcessParams, max_cells: int = MAX_CELLS_PER_AXIS,
             min_vox: float = 6.0) -> DepositionResult:
    """Run the shadow-evaporation engine.

    ``max_cells`` / ``min_vox`` set the ray-scan resolution: a larger
    ``max_cells`` (and smaller ``min_vox`` floor) traces the beam into a finer
    voxel grid — more accurate metal/junction edges at the cost of speed/memory.
    """
    boxes, R, z_top, resist_h, z_split = build_occluders(p)
    t_tot_metal = p.t_metal1 + p.t_metal2 + max(p.t_metal1 * 0.1, 3)

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
    lab = _label_solid(xs, ys, zs, boxes)

    # Both modes drive the two evaporations from their own (θ, φ).  Dolan
    # defaults to a uniaxial ±θ tilt at φ=0; Manhattan defaults to two beams at
    # the same tilt but orthogonal azimuth (φ₁=0, φ₂=90).  All four are free.
    d1 = beam_direction(p.angle1, p.phi1)
    d2 = beam_direction(p.angle2, p.phi2)

    al1 = _deposit(lab, xs, ys, zs, vox, d1, p.t_metal1, boxes)

    # Oxide: a thin (~few-nm) conformal skin coating ALL exposed faces of al1
    # — top and sides.  Physically it is far thinner than a voxel, so it is
    # drawn one cell thick for visibility but is NOT treated as geometry: it
    # does not shadow or block evaporation 2.
    neigh = np.zeros_like(al1)
    neigh[1:, :, :] |= al1[:-1, :, :]; neigh[:-1, :, :] |= al1[1:, :, :]
    neigh[:, 1:, :] |= al1[:, :-1, :]; neigh[:, :-1, :] |= al1[:, 1:, :]
    neigh[:, :, 1:] |= al1[:, :, :-1]; neigh[:, :, :-1] |= al1[:, :, 1:]
    alox = neigh & (~al1) & (lab == EMPTY)

    # evaporation 2 sees only the al1 metal as additional solid (oxide is
    # geometrically negligible and does not occlude the beam)
    lab2 = lab.copy()
    lab2[al1] = RESIST
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
    meta = dict(R=R, grid_R=grid_R, z_top=z_top, resist_h=resist_h, vox=vox,
                d1=d1, d2=d2, z_floor=z_floor_v, z_split=z_split,
                junc_xmax=junc_xmax, junc_ymax=junc_ymax,
                max_cells=max_cells, min_vox=min_vox)
    return DepositionResult(xs, ys, zs, vox, lab, al1, al2, alox, z_top, meta)


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


def junction_footprint(r: DepositionResult, min_cells: int = 2):
    """xy cells (on the substrate floor) where Al1 and Al2 stack up.

    Returns ``(mask_xy, area_nm2, ox_nm, oy_nm, juncs)``.  A single device can
    contain several distinct Josephson junctions (e.g. one on each side of a
    Dolan bridge for some tilt angles); each spatially separate Al1∩Al2 overlap
    is reported as its own entry in ``juncs`` — a list of dicts with keys
    ``mask, area, ox, oy, cx, cy, cells`` (sorted largest-area first).

    ``area`` is the total over all junctions; the headline ``ox/oy`` are those
    of the largest junction (not a bounding box merging separate junctions).
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

    # split into spatially separate junctions (drop sub-`min_cells` specks)
    labels, n = _connected_components(junc)
    juncs = []
    clean = np.zeros_like(junc)
    for lab_id in range(1, n + 1):
        comp = labels == lab_id
        cnt = int(comp.sum())
        if cnt < min_cells:
            continue
        clean |= comp
        ci, cj = np.where(comp)
        ox_c = (ci.max() - ci.min() + 1) * r.vox
        oy_c = (cj.max() - cj.min() + 1) * r.vox
        juncs.append(dict(
            mask=comp, cells=cnt, area=cnt * r.vox * r.vox,
            ox=ox_c, oy=oy_c,
            cx=float(r.xs[ci].mean()), cy=float(r.ys[cj].mean()),
        ))
    juncs.sort(key=lambda d: -d["area"])

    junc = clean
    area = sum(d["area"] for d in juncs)
    if juncs:
        ox, oy = juncs[0]["ox"], juncs[0]["oy"]
    else:
        ox = oy = 0.0
    return junc, area, ox, oy, juncs
