"""
voxel_view.py
=============
Render cross sections and top-view floor maps directly from the physical voxel
deposition result produced by deposition3d.simulate().

Both planes are supported:
  * x–z slice at a chosen y   (the classic evaporation cross section)
  * y–z slice at a chosen x   (the perpendicular cross section)

The bilayer resist is shown as two distinct layers (lower undercut sublayer vs
upper imaging resist), split at meta['z_split'].

Staged views (resist → evap1 → oxidation → evap2 → lift-off) are available for
both the cross section (render_stages) and the top view (render_top_stages),
with the Josephson junction (the AlOx overlap) highlighted in lift-off.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from deposition3d import EMPTY, RESIST, SUBSTRATE, DepositionResult


# ── category codes (drawing priority = numeric order, high wins) ───
C_EMPTY = 0
C_SUBSTRATE = 1
C_RESIST_LO = 2     # lower resist (undercut sublayer: MMA / undercut layer)
C_RESIST_UP = 3     # upper resist (imaging layer: PMMA / thick resist)
C_AL1 = 4
C_AL2 = 5
C_ALOX = 6
C_JUNC = 7

_COLORS = {
    C_EMPTY:     "#ffffff",
    C_SUBSTRATE: "#cfcfcf",
    C_RESIST_LO: "#f4e7c3",   # pale (undercut)
    C_RESIST_UP: "#d9b25a",   # darker (imaging resist)
    C_AL1:       "#5fa8d3",
    C_AL2:       "#ef767a",
    C_ALOX:      "#8338ec",
    C_JUNC:      "#2a9d8f",
}
_LABELS = {
    C_EMPTY:     "empty",
    C_SUBSTRATE: "substrate",
    C_RESIST_LO: "resist lower (undercut)",
    C_RESIST_UP: "resist upper (imaging)",
    C_AL1:       "Al #1",
    C_AL2:       "Al #2",
    C_ALOX:      "AlOx",
    C_JUNC:      "junction (Al1∩Al2)",
}

_N = 8
_CMAP = ListedColormap([_COLORS[i] for i in range(_N)])
_NORM = BoundaryNorm(np.arange(-0.5, _N + 0.5, 1.0), _CMAP.N)


# ════════════════════════════════════════════════════════════════
# low-level drawing helpers
# ════════════════════════════════════════════════════════════════

def _draw(ax, cat, h_axis, v_axis, h_label, v_label, title, hlim=None):
    """imshow a category grid; cat has shape (n_h, n_v) → transpose for display."""
    extent = [h_axis[0], h_axis[-1], v_axis[0], v_axis[-1]]
    ax.imshow(cat.T, origin="lower", extent=extent, aspect="auto",
              cmap=_CMAP, norm=_NORM, interpolation="nearest")
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    if hlim is not None:
        ax.set_xlim(-hlim, hlim)


def _legend(fig, present):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_COLORS[c]) for c in present]
    labels = [_LABELS[c] for c in present]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(present), 4),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))


def _beam_arrow_cs(ax, d, plane, hlim, ztop, color, label, side=-1):
    """Draw the evaporation beam direction as an arrow on a cross section."""
    dh = d[0] if plane == "x-z" else d[1]
    dz = d[2]
    n = np.hypot(dh, dz) or 1.0
    dh, dz = dh / n, dz / n
    L = 0.6 * ztop
    head = np.array([side * 0.45 * hlim, 0.45 * ztop])   # near surface
    tail = head - L * np.array([dh, dz])                 # up toward source
    ax.annotate("", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
    ax.text(tail[0], tail[1] + 0.06 * ztop, label, color=color,
            fontsize=8, ha="center", fontweight="bold")


def _beam_arrow_top(ax, d, hw, color, label, side=-1):
    """Draw the in-plane (azimuth) beam direction as an arrow on the top view."""
    dh, dv = d[0], d[1]
    n = np.hypot(dh, dv)
    if n < 1e-9:
        return
    dh, dv = dh / n, dv / n
    L = 0.5 * hw
    head = np.array([0.0, 0.0])
    tail = head - L * np.array([dh, dv])
    ax.annotate("", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
    ax.text(tail[0], tail[1], label, color=color, fontsize=8, fontweight="bold")


def _zoom_half(r, plane):
    """Half-width [nm] of a view window centred on the junction."""
    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    half = (jxm if plane == "x-z" else jym)
    return min(max(half * 2.5, 300.0), r.meta["R"])


# ════════════════════════════════════════════════════════════════
# cross-section slices
# ════════════════════════════════════════════════════════════════

def _slice_planes(r, plane, slice_pos):
    """Return (solid2d, al1, al2, alox, h_axis, h_label) for one slice."""
    if plane == "x-z":
        j = r.idx_y(slice_pos)
        return (r.solid[:, j, :], r.al1[:, j, :], r.al2[:, j, :],
                r.alox[:, j, :], r.xs, "x  [nm]  (evaporation axis)")
    i = r.idx_x(slice_pos)
    return (r.solid[i, :, :], r.al1[i, :, :], r.al2[i, :, :],
            r.alox[i, :, :], r.ys, "y  [nm]")


def _resist_cat_cross(cat, solid2d, zs, z_split):
    """Paint lower/upper resist into a cross-section category grid."""
    res = solid2d == RESIST
    lower_z = (zs < z_split)[None, :]
    cat[res & lower_z] = C_RESIST_LO
    cat[res & ~lower_z] = C_RESIST_UP


def render_cross_section(r: DepositionResult, plane="x-z", slice_pos=0.0,
                         junc_mask=None):
    """Render one combined cross-section slice (all layers, junction marked)."""
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    z_split = r.meta.get("z_split", r.z_top)
    solid2d, al1, al2, alox, h_axis, h_label = _slice_planes(r, plane, slice_pos)

    cat = np.full(solid2d.shape, C_EMPTY, np.int8)
    cat[solid2d == SUBSTRATE] = C_SUBSTRATE
    _resist_cat_cross(cat, solid2d, zs, z_split)
    cat[al1] = C_AL1
    cat[al2] = C_AL2
    cat[alox] = C_ALOX
    if junc_mask is not None:
        jc = junc_mask[:, r.idx_y(slice_pos)] if plane == "x-z" \
            else junc_mask[r.idx_x(slice_pos)]
        junc2 = jc[:, None] & (zs[None, :] >= 0) & (zs[None, :] < zf)
        cat[junc2] = C_JUNC

    title = (f"x–z cross section  (y = {slice_pos:.0f} nm)" if plane == "x-z"
             else f"y–z cross section  (x = {slice_pos:.0f} nm)")
    hlim = _zoom_half(r, plane)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title, hlim=hlim)
    _beam_arrow_cs(ax, r.meta["d1"], plane, hlim, r.z_top, _COLORS[C_AL1],
                   "evap 1", side=-1)
    _beam_arrow_cs(ax, r.meta["d2"], plane, hlim, r.z_top, _COLORS[C_AL2],
                   "evap 2", side=+1)
    _legend(fig, sorted(np.unique(cat).tolist()))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def render_stages(r: DepositionResult, plane="x-z", slice_pos=0.0, junc_mask=None):
    """5-panel staged cross section: resist → evap1 → oxidation → evap2 → lift-off."""
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    z_split = r.meta.get("z_split", r.z_top)
    solid2d, al1, al2, alox, h_axis, h_label = _slice_planes(r, plane, slice_pos)

    if junc_mask is not None:
        jc = junc_mask[:, r.idx_y(slice_pos)] if plane == "x-z" \
            else junc_mask[r.idx_x(slice_pos)]
        junc2 = jc[:, None] & (zs[None, :] >= 0) & (zs[None, :] < zf)
    else:
        junc2 = None

    sub = solid2d == SUBSTRATE

    def cat_build(inc_al1=False, inc_alox=False, inc_al2=False,
                  liftoff=False, emphasise_junc=False):
        cat = np.full(solid2d.shape, C_EMPTY, np.int8)
        cat[sub] = C_SUBSTRATE
        if not liftoff:
            _resist_cat_cross(cat, solid2d, zs, z_split)
            if inc_al1: cat[al1] = C_AL1
            if inc_alox: cat[alox] = C_ALOX
            if inc_al2: cat[al2] = C_AL2
        else:
            metal_any = al1 | al2 | alox
            floor = (zs >= 0) & (zs < zf)
            surv = (metal_any & floor[None, :]).any(axis=1)
            keep = surv[:, None]
            cat[al1 & keep] = C_AL1
            cat[al2 & keep] = C_AL2
            cat[alox & keep] = C_ALOX
            if emphasise_junc and junc2 is not None:
                cat[junc2 & keep] = C_JUNC
        return cat

    panels = [
        ("1. Resist only",   cat_build()),
        ("2. Evaporation 1", cat_build(inc_al1=True)),
        ("3. Oxidation",     cat_build(inc_al1=True, inc_alox=True)),
        ("4. Evaporation 2", cat_build(inc_al1=True, inc_alox=True, inc_al2=True)),
        ("5. Lift-off (JJ)", cat_build(liftoff=True, emphasise_junc=True)),
    ]

    hlim = _zoom_half(r, plane)
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4), sharey=True)
    for k, (ax, (title, cat)) in enumerate(zip(axes, panels)):
        _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title, hlim=hlim)
        if k in (1, 2):       # evap-1 related panels
            _beam_arrow_cs(ax, r.meta["d1"], plane, hlim, r.z_top,
                           _COLORS[C_AL1], "evap 1", side=-1)
        if k == 3:            # evap-2 panel
            _beam_arrow_cs(ax, r.meta["d2"], plane, hlim, r.z_top,
                           _COLORS[C_AL2], "evap 2", side=+1)
    _legend(fig, [C_SUBSTRATE, C_RESIST_LO, C_RESIST_UP,
                  C_AL1, C_ALOX, C_AL2, C_JUNC])
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ════════════════════════════════════════════════════════════════
# top view (floor map)
# ════════════════════════════════════════════════════════════════

def _floor_maps(r):
    """Return (al1f, al2f, aloxf, upper_resist, lower_resist) as (Nx,Ny) bools."""
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    z_split = r.meta.get("z_split", r.z_top)
    floor = (zs >= 0) & (zs < zf)
    al1f = r.al1[:, :, floor].any(axis=2)
    al2f = r.al2[:, :, floor].any(axis=2)
    aloxf = r.alox[:, :, floor].any(axis=2)
    lo_band = (zs >= 0) & (zs < z_split)
    up_band = (zs >= z_split) & (zs < r.z_top)
    lower_resist = (r.solid[:, :, lo_band] == RESIST).any(axis=2)
    upper_resist = (r.solid[:, :, up_band] == RESIST).any(axis=2)
    return al1f, al2f, aloxf, upper_resist, lower_resist


def _resist_cat_top(cat, upper_resist, lower_resist):
    """Paint the resist opening pattern (incl. undercut shelf) from above."""
    # solid resist (both layers) = upper imaging colour
    cat[upper_resist & lower_resist] = C_RESIST_UP
    # open below but resist above = the undercut shelf / overhang
    cat[upper_resist & ~lower_resist] = C_RESIST_LO
    # through-hole (open both) stays empty (substrate visible)


def render_top_stages(r: DepositionResult, junc_mask=None):
    """5-panel staged top view: resist (with undercut) → evap1 → ox → evap2 → lift-off."""
    al1f, al2f, aloxf, upper_resist, lower_resist = _floor_maps(r)

    def cat_build(inc_al1=False, inc_alox=False, inc_al2=False,
                  liftoff=False, emphasise_junc=False):
        cat = np.full(al1f.shape, C_EMPTY, np.int8)
        if not liftoff:
            _resist_cat_top(cat, upper_resist, lower_resist)
            if inc_al1: cat[al1f] = C_AL1
            if inc_alox: cat[aloxf] = C_ALOX
            if inc_al2: cat[al2f] = C_AL2
        else:
            cat[al1f] = C_AL1
            cat[al2f] = C_AL2
            cat[aloxf] = C_ALOX
            if emphasise_junc and junc_mask is not None:
                cat[junc_mask] = C_JUNC
        return cat

    panels = [
        ("1. Resist only",   cat_build()),
        ("2. Evaporation 1", cat_build(inc_al1=True)),
        ("3. Oxidation",     cat_build(inc_al1=True, inc_alox=True)),
        ("4. Evaporation 2", cat_build(inc_al1=True, inc_alox=True, inc_al2=True)),
        ("5. Lift-off (JJ)", cat_build(liftoff=True, emphasise_junc=True)),
    ]

    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    hw = min(max(jxm, jym) * 3.0, r.meta["R"])
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6), sharey=True)
    for k, (ax, (title, cat)) in enumerate(zip(axes, panels)):
        _draw(ax, cat, r.xs, r.ys, "x  [nm]", "y  [nm]", title)
        ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
        ax.set_aspect("equal")
        if k in (1, 2):
            _beam_arrow_top(ax, r.meta["d1"], hw, _COLORS[C_AL1], "evap 1")
        if k == 3:
            _beam_arrow_top(ax, r.meta["d2"], hw, _COLORS[C_AL2], "evap 2")
    _legend(fig, [C_RESIST_LO, C_RESIST_UP, C_AL1, C_ALOX, C_AL2, C_JUNC])
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def render_top_view(r: DepositionResult, junc_mask=None):
    """Single top-down floor map: Al1 / Al2 / junction on the substrate."""
    al1f, al2f, aloxf, upper_resist, lower_resist = _floor_maps(r)
    cat = np.full(al1f.shape, C_EMPTY, np.int8)
    _resist_cat_top(cat, upper_resist, lower_resist)
    cat[al1f] = C_AL1
    cat[al2f] = C_AL2
    cat[aloxf] = C_ALOX
    if junc_mask is not None:
        cat[junc_mask] = C_JUNC

    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    hw = min(max(jxm, jym) * 3.0, r.meta["R"])
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    _draw(ax, cat, r.xs, r.ys, "x  [nm]", "y  [nm]", "Top view (floor deposit)")
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
    ax.set_aspect("equal")
    _beam_arrow_top(ax, r.meta["d1"], hw, _COLORS[C_AL1], "evap 1")
    _beam_arrow_top(ax, r.meta["d2"], hw, _COLORS[C_AL2], "evap 2")
    present = sorted([c for c in np.unique(cat).tolist() if c != C_EMPTY])
    if present:
        _legend(fig, present)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig
