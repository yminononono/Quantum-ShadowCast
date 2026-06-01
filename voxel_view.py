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
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from deposition3d import EMPTY, RESIST, SUBSTRATE, DepositionResult

# ── global look & feel ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#0e1117",
    "savefig.facecolor": "#0e1117",
    "text.color":        "#e6e6e6",
    "axes.labelcolor":   "#c8c8c8",
    "axes.edgecolor":    "#3a3f4b",
    "xtick.color":       "#9aa0aa",
    "ytick.color":       "#9aa0aa",
    "axes.titlecolor":   "#f0f0f0",
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.labelsize":    9.5,
    "font.size":         9,
    "axes.grid":         False,
    "figure.dpi":        110,
})

_OX_LINE = "#c77dff"   # oxide outline colour
_OX_THICK = 3.0        # AlOx visual thickness [nm] (real barrier ≈ 1–2 nm)


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
    C_EMPTY:     "#0e1117",   # = background (invisible)
    C_SUBSTRATE: "#4a4f5a",
    C_RESIST_LO: "#7a6f57",   # pale (undercut sublayer)
    C_RESIST_UP: "#c7a15a",   # imaging resist
    C_AL1:       "#4cc9f0",
    C_AL2:       "#f72585",
    C_ALOX:      _OX_LINE,
    C_JUNC:      "#2ce0b3",
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

def _axis_edges(axis):
    """Cell-edge bounds (lo, hi) for an axis of cell *centres*.

    imshow's ``extent`` must span the outer cell *edges*, not the centres —
    using centres shrinks/shifts every cell by half a voxel, which is what made
    the substrate top, deposited metal and oxide skin look misaligned.  Because
    the z grid is built as ``arange(-vox/2, …)`` the substrate cell centre is at
    -vox/2, so its top edge lands exactly at z = 0.
    """
    a = np.asarray(axis, float)
    if a.size < 2:
        return float(a[0]) - 0.5, float(a[0]) + 0.5
    return float(a[0] - (a[1] - a[0]) / 2.0), float(a[-1] + (a[-1] - a[-2]) / 2.0)


def _extent(h_axis, v_axis):
    hlo, hhi = _axis_edges(h_axis)
    vlo, vhi = _axis_edges(v_axis)
    return [hlo, hhi, vlo, vhi]


def _draw(ax, cat, h_axis, v_axis, h_label, v_label, title, hlim=None, vlim=None):
    """imshow a category grid; cat has shape (n_h, n_v) → transpose for display."""
    extent = _extent(h_axis, v_axis)
    ax.imshow(cat.T, origin="lower", extent=extent, aspect="auto",
              cmap=_CMAP, norm=_NORM, interpolation="nearest", zorder=1)
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title, pad=8)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.tick_params(length=3, width=0.7)
    if hlim is not None:
        ax.set_xlim(-hlim, hlim)
    if vlim is not None:
        ax.set_ylim(vlim[0], vlim[1])


def _overlay_cats(ax, cat, h_axis, v_axis, cats, alpha=0.62, zorder=3):
    """Overlay selected categories as a translucent RGBA image.

    Lets whatever is drawn beneath (resist / undercut shelf) remain visible
    through the deposited metal in the top view.
    """
    nh, nv = cat.shape
    rgba = np.zeros((nh, nv, 4), dtype=float)
    any_set = False
    for c in cats:
        m = cat == c
        if m.any():
            rgba[m] = to_rgba(_COLORS[c], alpha)
            any_set = True
    if not any_set:
        return
    extent = _extent(h_axis, v_axis)
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", extent=extent,
              aspect="auto", interpolation="nearest", zorder=zorder)


def _oxide_edges_cs(ax, al1, exclude, h_axis, v_axis, ox_t=_OX_THICK, zorder=6):
    """Draw the AlOx barrier as a thin (≈few-nm) skin on the exposed Al1 faces.

    For every Al1 cell, each of its 4 in-plane neighbours that is *oxidizable*
    (i.e. air or Al2 — anything not in ``exclude`` and not Al1) gets a thin
    ``ox_t``-wide stroke laid on that face.  Perpendicular faces are extended by
    ``ox_t`` so convex CORNERS fill in (oxide wraps the metal corner).  Because
    Al2-facing faces are oxidizable, the barrier is drawn at the Al1/Al2
    interface too — so Al2 (its filled imshow cell) reads as sitting on top of
    the thin oxide rather than the oxide being omitted there.
    """
    if not al1.any():
        return
    nh, nv = al1.shape
    dh = float(h_axis[1] - h_axis[0]) if nh > 1 else 1.0
    dv = float(v_axis[1] - v_axis[0]) if nv > 1 else 1.0
    hh, hv = dh / 2.0, dv / 2.0
    oxidizable = ~(exclude | al1)          # air OR Al2 (everything Al1 can oxidize against)
    ii, kk = np.where(al1)
    rects = []
    for i, k in zip(ii.tolist(), kk.tolist()):
        x, z = float(h_axis[i]), float(v_axis[k])
        if k + 1 < nv and oxidizable[i, k + 1]:
            rects.append(Rectangle((x - hh - ox_t, z + hv), dh + 2 * ox_t, ox_t))
        if k - 1 >= 0 and oxidizable[i, k - 1]:
            rects.append(Rectangle((x - hh - ox_t, z - hv - ox_t), dh + 2 * ox_t, ox_t))
        if i + 1 < nh and oxidizable[i + 1, k]:
            rects.append(Rectangle((x + hh, z - hv - ox_t), ox_t, dv + 2 * ox_t))
        if i - 1 >= 0 and oxidizable[i - 1, k]:
            rects.append(Rectangle((x - hh - ox_t, z - hv - ox_t), ox_t, dv + 2 * ox_t))
    if rects:
        ax.add_collection(PatchCollection(rects, facecolor=_OX_LINE,
                                          edgecolor="none", zorder=zorder))


def _oxide_fill(ax, ox_mask, h_axis, v_axis, alpha=0.9, zorder=5):
    """Fill the AlOx cells as solid squares sitting on the metal."""
    if not ox_mask.any():
        return
    cat = np.where(ox_mask, np.int8(C_ALOX), np.int8(C_EMPTY))
    _overlay_cats(ax, cat, h_axis, v_axis, [C_ALOX], alpha=alpha, zorder=zorder)


def _legend(fig, present, oxide=True):
    handles, labels = [], []
    for c in present:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=_COLORS[c], ec="none"))
        labels.append(_LABELS[c])
    if oxide:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=_OX_LINE, ec="none",
                                      alpha=0.9))
        labels.append(_LABELS[C_ALOX])
    leg = fig.legend(handles, labels, loc="lower center",
                     ncol=min(len(handles), 5), frameon=False, fontsize=8,
                     bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts():
        t.set_color("#d0d0d0")


def _junc_labels(ax, juncs):
    """Annotate each separate Josephson junction with J1, J2, … at its centre."""
    if not juncs or len(juncs) < 2:
        return
    for n, jd in enumerate(juncs, 1):
        ax.text(jd["cx"], jd["cy"], f"J{n}", color="#0e1117",
                fontsize=8.5, fontweight="bold", ha="center", va="center",
                zorder=8,
                bbox=dict(boxstyle="circle,pad=0.18", fc=_COLORS[C_JUNC],
                          ec="#0e1117", lw=0.8))


def _beam_arrow_cs(ax, d, plane, hlim, ztop, color, label, side=-1):
    """Draw the evaporation beam direction as an arrow on a cross section."""
    dh = d[0] if plane == "x-z" else d[1]
    dz = d[2]
    n = np.hypot(dh, dz) or 1.0
    dh, dz = dh / n, dz / n
    L = 0.42 * ztop
    head = np.array([side * 0.42 * hlim, 0.34 * ztop])   # near surface
    tail = head - L * np.array([dh, dz])                 # up toward source
    ax.annotate("", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
    ly = min(tail[1] + 0.04 * ztop, 0.93 * ztop)         # keep label inside axes
    ax.text(tail[0], ly, label, color=color,
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


def _zoom_half(r, plane, view_half=None):
    """Half-width [nm] of a view window centred on the junction.

    When ``view_half`` is given it overrides the auto window (clamped to the
    simulated grid extent); otherwise the window auto-fits the junction zone.
    """
    grid_R = r.meta.get("grid_R", r.meta["R"])
    if view_half is not None:
        return float(np.clip(view_half, 50.0, grid_R))
    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    half = (jxm if plane == "x-z" else jym)
    return min(max(half * 2.5, 300.0), grid_R)


def _top_half(r, view_half=None):
    """Half-width [nm] of the square top-view window (override-aware)."""
    grid_R = r.meta.get("grid_R", r.meta["R"])
    if view_half is not None:
        return float(np.clip(view_half, 50.0, grid_R))
    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    return min(max(jxm, jym) * 3.0, grid_R)


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
                         junc_mask=None, view_half=None, zmax=None):
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
    if junc_mask is not None:
        jc = junc_mask[:, r.idx_y(slice_pos)] if plane == "x-z" \
            else junc_mask[r.idx_x(slice_pos)]
        junc2 = jc[:, None] & (zs[None, :] >= 0) & (zs[None, :] < zf)
        cat[junc2] = C_JUNC

    title = (f"x–z cross section  (y = {slice_pos:.0f} nm)" if plane == "x-z"
             else f"y–z cross section  (x = {slice_pos:.0f} nm)")
    hlim = _zoom_half(r, plane, view_half)
    vlim = (zs[0], zmax) if zmax is not None else None
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title, hlim=hlim, vlim=vlim)
    vtop = zmax if zmax is not None else float(zs[-1])
    # AlOx: a thin (~3 nm) skin on every exposed Al1 face, incl. the Al1/Al2
    # interface (so Al2 sits on top of the barrier) and the metal corners.
    _oxide_edges_cs(ax, al1, (solid2d == RESIST) | (solid2d == SUBSTRATE),
                    h_axis, zs)
    _beam_arrow_cs(ax, r.meta["d1"], plane, hlim, vtop, _COLORS[C_AL1],
                   "evap 1", side=-1)
    _beam_arrow_cs(ax, r.meta["d2"], plane, hlim, vtop, _COLORS[C_AL2],
                   "evap 2", side=+1)
    present = [c for c in sorted(np.unique(cat).tolist()) if c != C_EMPTY]
    _legend(fig, present)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def render_stages(r: DepositionResult, plane="x-z", slice_pos=0.0, junc_mask=None,
                  view_half=None, zmax=None):
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

    # grounded metal (for lift-off): contiguous metal stack from the floor up.
    metal_any = al1 | al2
    z0 = int(np.searchsorted(zs, 0.0))
    grounded = np.zeros_like(metal_any)
    grounded[:, z0:] = np.cumprod(metal_any[:, z0:].astype(np.int8),
                                  axis=1).astype(bool)

    def cat_build(inc_al1=False, inc_al2=False,
                  liftoff=False, emphasise_junc=False):
        cat = np.full(solid2d.shape, C_EMPTY, np.int8)
        cat[sub] = C_SUBSTRATE
        if not liftoff:
            _resist_cat_cross(cat, solid2d, zs, z_split)
            if inc_al1: cat[al1] = C_AL1
            if inc_al2: cat[al2] = C_AL2
        else:
            # Lift-off: strip the resist; only grounded metal survives.
            cat[al1 & grounded] = C_AL1
            cat[al2 & grounded] = C_AL2
            if emphasise_junc and junc2 is not None:
                cat[junc2 & grounded] = C_JUNC
        return cat

    # oxide appears once Al1 is present and stays through evap-2 / lift-off.
    # Drawn as a thin skin on the exposed Al1 faces: before lift-off the resist
    # (and substrate) block oxidation; at lift-off the resist is gone so only
    # the surviving (grounded) Al1 oxidises against air / Al2.
    ox_excl_pre = (solid2d == RESIST) | sub
    panels = [
        ("1. Resist only",   cat_build(),                                False),
        ("2. Evaporation 1", cat_build(inc_al1=True),                    False),
        ("3. Oxidation",     cat_build(inc_al1=True),                    True),
        ("4. Evaporation 2", cat_build(inc_al1=True, inc_al2=True),      True),
        ("5. Lift-off (JJ)", cat_build(liftoff=True, emphasise_junc=True), True),
    ]

    hlim = _zoom_half(r, plane, view_half)
    vlim = (zs[0], zmax) if zmax is not None else None
    vtop = zmax if zmax is not None else float(zs[-1])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for k, (ax, (title, cat, show_ox)) in enumerate(zip(axes, panels)):
        _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title, hlim=hlim, vlim=vlim)
        if show_ox:
            if k == 4:        # lift-off: resist stripped, grounded Al1 only
                _oxide_edges_cs(ax, al1 & grounded, sub, h_axis, zs)
            else:
                _oxide_edges_cs(ax, al1, ox_excl_pre, h_axis, zs)
        if k in (1, 2):       # evap-1 related panels
            _beam_arrow_cs(ax, r.meta["d1"], plane, hlim, vtop,
                           _COLORS[C_AL1], "evap 1", side=-1)
        if k == 3:            # evap-2 panel
            _beam_arrow_cs(ax, r.meta["d2"], plane, hlim, vtop,
                           _COLORS[C_AL2], "evap 2", side=+1)
    axes[-1].axis("off")      # 6th cell unused (5 stages)
    _legend(fig, [C_SUBSTRATE, C_RESIST_LO, C_RESIST_UP,
                  C_AL1, C_AL2, C_JUNC])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
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


def render_top_stages(r: DepositionResult, junc_mask=None, view_half=None,
                      juncs=None):
    """5-panel staged top view: resist (with undercut) → evap1 → ox → evap2 → lift-off.

    The resist / undercut pattern is drawn opaque underneath, then the deposited
    metal is layered on top semi-transparently so the resist mask stays visible.
    The AlOx barrier is filled (as squares) on top of the Al1 footprint, and any
    separate Josephson junctions are labelled J1, J2, … in the lift-off panel.
    """
    al1f, al2f, aloxf, upper_resist, lower_resist = _floor_maps(r)

    # Opaque resist base (substrate visible through the open holes).
    resist_base = np.full(al1f.shape, C_EMPTY, np.int8)
    _resist_cat_top(resist_base, upper_resist, lower_resist)
    empty_base = np.full(al1f.shape, C_EMPTY, np.int8)

    def metal_cat(inc_al1=False, inc_al2=False, emphasise_junc=False):
        cat = np.full(al1f.shape, C_EMPTY, np.int8)
        if inc_al1: cat[al1f] = C_AL1
        if inc_al2: cat[al2f] = C_AL2
        if emphasise_junc and junc_mask is not None:
            cat[junc_mask] = C_JUNC
        return cat

    ox_pre = aloxf                       # oxidation: oxide over all Al1
    ox_post = aloxf & ~al2f              # after evap-2: Al2 sits over the barrier
    # (title, base_grid, metal_grid, metal_alpha, ox_mask, label_junc)
    panels = [
        ("1. Resist only",   resist_base, metal_cat(),                          0.62, None,    False),
        ("2. Evaporation 1", resist_base, metal_cat(inc_al1=True),              0.62, None,    False),
        ("3. Oxidation",     resist_base, metal_cat(inc_al1=True),              0.62, ox_pre,  False),
        ("4. Evaporation 2", resist_base, metal_cat(inc_al1=True, inc_al2=True),0.62, ox_post, False),
        ("5. Lift-off (JJ)", empty_base,
         metal_cat(inc_al1=True, inc_al2=True, emphasise_junc=True),            1.00, ox_post, True),
    ]

    hw = _top_half(r, view_half)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for k, (ax, (title, base, mcat, malpha, ox_mask, lbl)) in enumerate(zip(axes, panels)):
        _draw(ax, base, r.xs, r.ys, "x  [nm]", "y  [nm]", title)
        _overlay_cats(ax, mcat, r.xs, r.ys,
                      [C_AL1, C_AL2, C_JUNC], alpha=malpha, zorder=3)
        if ox_mask is not None:
            _oxide_fill(ax, ox_mask, r.xs, r.ys, alpha=0.5, zorder=4)
        ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
        ax.set_aspect("equal")
        if lbl:
            _junc_labels(ax, juncs)
        if k in (1, 2):
            _beam_arrow_top(ax, r.meta["d1"], hw, _COLORS[C_AL1], "evap 1")
        if k == 3:
            _beam_arrow_top(ax, r.meta["d2"], hw, _COLORS[C_AL2], "evap 2")
    axes[-1].axis("off")      # 6th cell unused (5 stages)
    _legend(fig, [C_RESIST_LO, C_RESIST_UP, C_AL1, C_AL2, C_JUNC])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def render_top_view(r: DepositionResult, junc_mask=None, view_half=None,
                    juncs=None):
    """Single top-down floor map: resist/undercut (opaque) with Al1 / Al2 /
    junction layered on top semi-transparently so the resist mask stays visible.
    The AlOx barrier is filled on top of Al1 and separate junctions are labelled."""
    al1f, al2f, aloxf, upper_resist, lower_resist = _floor_maps(r)
    base = np.full(al1f.shape, C_EMPTY, np.int8)
    _resist_cat_top(base, upper_resist, lower_resist)

    mcat = np.full(al1f.shape, C_EMPTY, np.int8)
    mcat[al1f] = C_AL1
    mcat[al2f] = C_AL2
    if junc_mask is not None:
        mcat[junc_mask] = C_JUNC

    hw = _top_half(r, view_half)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    _draw(ax, base, r.xs, r.ys, "x  [nm]", "y  [nm]", "Top view (floor deposit)")
    _overlay_cats(ax, mcat, r.xs, r.ys, [C_AL1, C_AL2, C_JUNC],
                  alpha=0.62, zorder=3)
    _oxide_fill(ax, aloxf & ~al2f, r.xs, r.ys, alpha=0.5, zorder=4)
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
    ax.set_aspect("equal")
    _junc_labels(ax, juncs)
    _beam_arrow_top(ax, r.meta["d1"], hw, _COLORS[C_AL1], "evap 1")
    _beam_arrow_top(ax, r.meta["d2"], hw, _COLORS[C_AL2], "evap 2")
    present = [c for c in (C_RESIST_LO, C_RESIST_UP) if (base == c).any()]
    present += [c for c in (C_AL1, C_AL2, C_JUNC) if (mcat == c).any()]
    if present:
        _legend(fig, present)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig
