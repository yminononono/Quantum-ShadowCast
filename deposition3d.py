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


def _deposit(lab, xs, ys, zs, vox, d, t_metal, boxes, occ_mask=None):
    """Return bool grid of metal voxels for one evaporation.

    A cell receives metal if it is EMPTY, its beam-forward neighbour is solid
    (it sits on a surface facing the beam), and the ray toward the source is
    unobstructed.  The film is then grown `n` cells back toward the source.

    ``occ_mask`` (optional bool grid) is extra occluding metal from a prior
    evaporation: a surface cell is also shadowed if its ray to the source passes
    through it — this is the sidewall effect (prior wall coating narrows the
    opening).  ``None`` ⇒ resist-only shadowing (unchanged behaviour).
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
    if occ_mask is not None:                 # sidewall: prior-metal wall coating shadows
        blocked = blocked | _occluded_mask(ii, jj, kk, d, occ_mask)
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
    lab = _label_solid(xs, ys, zs, boxes)

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
    if not trilayer:
        # ── Bilayer: evap1 → oxidation → evap2 ────────────────────
        d1 = beam_direction(p.angle1, p.phi1)
        d2 = beam_direction(p.angle2, p.phi2)
        al1 = _deposit(lab, xs, ys, zs, vox, d1, p.t_metal1, boxes)
        # Oxide: thin conformal skin on al1; NOT geometry (does not occlude).
        alox = _oxide_skin(al1)
        lab2 = lab.copy()
        lab2[al1] = RESIST
        al2 = _deposit(lab2, xs, ys, zs, vox, d2, p.t_metal2, boxes,
                       occ_mask=(al1 if p.sidewall else None))
        meta_d1, meta_d2 = d1, d2
    else:
        # ── Trilayer: Nb1 → Al2 → oxidation → Al3 → Nb4 ───────────
        # Electrode 1 (bottom) = Nb(evap1) + Al(evap2) at the Evap-1 angle;
        # electrode 2 (top)    = Al(evap3) + Nb(evap4) at the Evap-2 angle.
        d1 = beam_direction(p.angle1, p.phi1)        # evap1  Nb
        d2 = beam_direction(p.tri_angle2, p.tri_phi2)  # evap2  Al
        d3 = beam_direction(p.angle2, p.phi2)        # evap3  Al
        d4 = beam_direction(p.tri_angle4, p.tri_phi4)  # evap4  Nb

        nb1 = _deposit(lab, xs, ys, zs, vox, d1, p.tri_t1, boxes)
        lab_b = lab.copy(); lab_b[nb1] = RESIST
        al2f = _deposit(lab_b, xs, ys, zs, vox, d2, p.tri_t2, boxes,
                        occ_mask=(nb1 if p.sidewall else None))
        elec1 = nb1 | al2f                            # bottom electrode

        # Oxidation after evap1+evap2: skin on ALL exposed faces of the bottom
        # electrode — i.e. on both the Nb and the Al that face free space.
        alox = _oxide_skin(elec1)

        lab_c = lab.copy(); lab_c[elec1] = RESIST     # electrode 2 sees elec1 solid
        al3 = _deposit(lab_c, xs, ys, zs, vox, d3, p.tri_t3, boxes,
                       occ_mask=(elec1 if p.sidewall else None))
        lab_d = lab_c.copy(); lab_d[al3] = RESIST
        nb4 = _deposit(lab_d, xs, ys, zs, vox, d4, p.tri_t4, boxes,
                       occ_mask=((elec1 | al3) if p.sidewall else None))
        elec2 = al3 | nb4                             # top electrode

        al1, al2 = elec1, elec2                       # keep al1/al2 = the two electrodes
        films = dict(nb1=nb1, al2=al2f, al3=al3, nb4=nb4)
        meta_d1, meta_d2 = d1, d3                     # arrows: the two electrode beams
        tri_dirs = dict(nb1=d1, al2=d2, al3=d3, nb4=d4)  # per-evaporation beams

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
                max_cells=max_cells, min_vox=min_vox)
    return DepositionResult(xs, ys, zs, vox, lab, al1, al2, alox, z_top, meta,
                            stack="Trilayer" if trilayer else "Bilayer",
                            films=films)


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


def _junction_cells_3d(r: DepositionResult, min_cells: int = 2):
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


def junction_footprint(r: DepositionResult, min_cells: int = 2):
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
    clean, juncs = _junction_cells_3d(r, min_cells)
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


def junction_combos(r: DepositionResult, min_cells: int = 2):
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
    clean, _ = _junction_cells_3d(r, min_cells)

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
