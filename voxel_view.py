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
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from deposition3d import (EMPTY, RESIST, SUBSTRATE, DepositionResult,
                          COMBO_NBAL, COMBO_ALAL, COMBO_NBNB)
import process_engine as pe

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
# ── trilayer extras ───────────────────────────────────────────────
C_NB = 8            # legacy by-material Nb (superseded by C_E1_NB/C_E4_NB)
C_AL = 9            # legacy by-material Al (superseded by C_E2_AL/C_E3_AL)
C_JUNC_NBAL = 10    # junction barrier between Nb and Al
C_JUNC_ALAL = 11    # junction barrier between Al and Al
C_JUNC_NBNB = 12    # junction barrier between Nb and Nb
# per-evaporation metal shades: the film-level views distinguish the SAME metal
# by which evaporation deposited it, via brightness — evap1 Nb (dark amber) →
# evap4 Nb (bright amber); evap2 Al (dark blue) → evap3 Al (light blue).
C_E1_NB = 13        # evaporation 1 — Nb (electrode 1, lower)
C_E2_AL = 14        # evaporation 2 — Al (electrode 1, upper)
C_E3_AL = 15        # evaporation 3 — Al (electrode 2, lower)
C_E4_NB = 16        # evaporation 4 — Nb (electrode 2, upper)

_COLORS = {
    C_EMPTY:     "#0e1117",   # = background (invisible)
    C_SUBSTRATE: "#4a4f5a",
    C_RESIST_LO: "#7a6f57",   # pale (undercut sublayer)
    C_RESIST_UP: "#c7a15a",   # imaging resist
    C_AL1:       "#4cc9f0",
    C_AL2:       "#f72585",
    C_ALOX:      _OX_LINE,
    C_JUNC:      "#2ce0b3",
    C_NB:        "#ffb703",   # niobium (amber)
    C_AL:        "#8ecae6",   # aluminium (light blue)
    C_JUNC_NBAL: "#ffd166",   # Nb–Al overlap (gold)
    C_JUNC_ALAL: "#2ce0b3",   # Al–Al overlap (teal)
    C_JUNC_NBNB: "#ef476f",   # Nb–Nb overlap (raspberry)
    C_E1_NB:     "#a06e0a",   # evap-1 Nb (dark amber)
    C_E2_AL:     "#3c8cb4",   # evap-2 Al (dark blue)
    C_E3_AL:     "#bee4f5",   # evap-3 Al (light blue)
    C_E4_NB:     "#ffcd5a",   # evap-4 Nb (bright amber)
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
    C_NB:        "Nb",
    C_AL:        "Al",
    C_JUNC_NBAL: "junction Nb–Al",
    C_JUNC_ALAL: "junction Al–Al",
    C_JUNC_NBNB: "junction Nb–Nb",
    C_E1_NB:     "Nb (evap 1)",
    C_E2_AL:     "Al (evap 2)",
    C_E3_AL:     "Al (evap 3)",
    C_E4_NB:     "Nb (evap 4)",
}

# combo-code → junction category, for trilayer overlay colouring
_COMBO_CAT = {COMBO_NBAL: C_JUNC_NBAL, COMBO_ALAL: C_JUNC_ALAL,
              COMBO_NBNB: C_JUNC_NBNB}

_N = 17
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
        if np.isscalar(hlim):
            ax.set_xlim(-hlim, hlim)                      # symmetric half-width
        else:
            ax.set_xlim(hlim[0], hlim[1])                # explicit (lo, hi) window
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


def _beam_arrow_cs(ax, d, uvec, hlim, ztop, color, label, side=-1, xcenter=0.0):
    """Draw the evaporation beam direction as an arrow on a cross section.

    ``uvec=(ux,uy)`` is the in-plane unit vector of the slice direction; the
    beam's horizontal component is projected onto it so the arrow tilt is
    correct for an obliquely-oriented slice.  ``xcenter`` shifts the arrow with a
    panned horizontal window."""
    ux, uy = uvec
    dh = d[0] * ux + d[1] * uy
    dz = d[2]
    n = np.hypot(dh, dz) or 1.0
    dh, dz = dh / n, dz / n
    L = 0.42 * ztop
    head = np.array([xcenter + side * 0.42 * hlim, 0.34 * ztop])   # near surface
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


def _zoom_half(r, view_half=None):
    """Half-width [nm] of a view window centred on the junction.

    When ``view_half`` is given it overrides the auto window (clamped to the
    simulated grid extent); otherwise the window auto-fits the junction zone.
    The slice can be cut at an arbitrary in-plane angle, so the auto window
    uses the larger of the x / y junction extents.
    """
    grid_R = r.meta.get("grid_R", r.meta["R"])
    if view_half is not None:
        return float(np.clip(view_half, 50.0, grid_R))
    jxm = r.meta.get("junc_xmax", r.meta["R"])
    jym = r.meta.get("junc_ymax", r.meta["R"])
    half = max(jxm, jym)
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

def _slice_dirs(angle_deg):
    """In-plane unit vectors for a slice at azimuth ``angle_deg``.

    Returns ``(u, n)`` where ``u`` is along the slice and ``n`` is the
    perpendicular (offset) direction.  ``angle_deg=0`` → cut along x at
    y=offset; ``angle_deg=90`` → cut along y."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return (ca, sa), (-sa, ca)


def _oblique_columns(r, angle_deg, offset):
    """Sample voxel columns along an in-plane line at azimuth ``angle_deg`` with
    perpendicular ``offset`` [nm].

    Returns ``(ix, iy, s_axis, inside)``: nearest-neighbour column indices into
    the (Nx,Ny) grid for each point P(s) = s·u + offset·n, the running distance
    ``s`` along the slice, and a boolean ``inside`` flagging samples that fall
    within the simulated grid (the rest are clipped to the edge)."""
    (ux, uy), (nx, ny) = _slice_dirs(angle_deg)
    vox = r.vox
    grid_R = r.meta.get("grid_R", r.meta["R"])
    s = np.arange(-grid_R, grid_R + vox, vox)
    px = s * ux + offset * nx
    py = s * uy + offset * ny
    ix = np.rint((px - r.xs[0]) / vox).astype(int)
    iy = np.rint((py - r.ys[0]) / vox).astype(int)
    inside = ((ix >= 0) & (ix < len(r.xs)) &
              (iy >= 0) & (iy < len(r.ys)))
    ix = np.clip(ix, 0, len(r.xs) - 1)
    iy = np.clip(iy, 0, len(r.ys) - 1)
    return ix, iy, s, inside


def _slice_planes(r, angle_deg, offset):
    """Return (solid2d, al1, al2, alox, h_axis, h_label, ix, iy, films2d) for an
    oblique slice at azimuth ``angle_deg`` and perpendicular ``offset`` [nm].

    ``films2d`` is a dict {nb1, al2, al3, nb4} of (n_h, Nz) bool slices for a
    trilayer result (so the cross section can be coloured by material), else
    ``None`` for a bilayer."""
    ix, iy, s, inside = _oblique_columns(r, angle_deg, offset)
    solid2d = r.solid[ix, iy, :].copy()
    al1 = r.al1[ix, iy, :].copy()
    al2 = r.al2[ix, iy, :].copy()
    alox = r.alox[ix, iy, :].copy()
    out = ~inside
    if out.any():                         # blank columns outside the grid
        solid2d[out, :] = EMPTY
        al1[out, :] = False
        al2[out, :] = False
        alox[out, :] = False
    films2d = None
    if getattr(r, "stack", "Bilayer") == "Trilayer" and r.films:
        films2d = {}
        for k, g in r.films.items():
            sl = g[ix, iy, :].copy()
            if out.any():
                sl[out, :] = False
            films2d[k] = sl
    label = f"distance along slice  [nm]   (α = {angle_deg:.0f}°)"
    return solid2d, al1, al2, alox, s, label, ix, iy, films2d


def _paint_trilayer_metal(cat, films2d):
    """Colour trilayer metal cells by evaporation (per-film brightness shade).

    The same metal is distinguished by which evaporation laid it down: evap1 Nb
    (dark) vs evap4 Nb (bright); evap2 Al (dark) vs evap3 Al (light).  Later
    sublayers overwrite earlier ones where they share a cell, so the
    physically-upper film wins (matches the deposition sequence)."""
    cat[films2d["nb1"]] = C_E1_NB    # electrode-1 lower (evap 1, Nb)
    cat[films2d["al2"]] = C_E2_AL  # electrode-1 upper (evap 2, Al)
    cat[films2d["al3"]] = C_E3_AL    # electrode-2 lower (evap 3, Al)
    cat[films2d["nb4"]] = C_E4_NB    # electrode-2 upper (evap 4, Nb)


def _resist_cat_cross(cat, solid2d, zs, z_split):
    """Paint lower/upper resist into a cross-section category grid."""
    res = solid2d == RESIST
    lower_z = (zs < z_split)[None, :]
    cat[res & lower_z] = C_RESIST_LO
    cat[res & ~lower_z] = C_RESIST_UP


def render_cross_section(r: DepositionResult, angle_deg=0.0, offset=0.0,
                         junc_mask=None, view_half=None, zmax=None,
                         view_center=0.0, zmin=None):
    """Render one combined cross-section slice (all layers, junction marked).

    The slice is taken at in-plane azimuth ``angle_deg`` with perpendicular
    ``offset`` [nm] (``angle_deg=0`` → x–z cut at y=offset)."""
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    z_split = r.meta.get("z_split", r.z_top)
    solid2d, al1, al2, alox, h_axis, h_label, ix, iy, films2d = _slice_planes(
        r, angle_deg, offset)

    cat = np.full(solid2d.shape, C_EMPTY, np.int8)
    cat[solid2d == SUBSTRATE] = C_SUBSTRATE
    _resist_cat_cross(cat, solid2d, zs, z_split)
    if films2d is not None:               # trilayer: colour by material (Nb/Al)
        _paint_trilayer_metal(cat, films2d)
    else:
        cat[al1] = C_AL1
        cat[al2] = C_AL2
    if junc_mask is not None and films2d is None:
        # Bilayer: highlight the Al1∩Al2 overlap column.  For trilayer the
        # material colouring + oxide barrier already show the junction stack,
        # and the Nb-Al/Al-Al/Nb-Nb split lives in the top / junction views.
        jc = junc_mask[ix, iy]
        junc2 = jc[:, None] & (zs[None, :] >= 0) & (zs[None, :] < zf)
        cat[junc2] = C_JUNC

    title = f"Cross section  (α = {angle_deg:.0f}°,  offset = {offset:.0f} nm)"
    (ux, uy), _ = _slice_dirs(angle_deg)
    half = _zoom_half(r, view_half)
    hwin = (view_center - half, view_center + half)
    vlo = float(zs[0] if zmin is None else zmin)
    vhi = float(zs[-1] if zmax is None else zmax)
    vlim = (vlo, vhi); vtop = vhi
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=140)
    _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title, hlim=hwin, vlim=vlim)
    # AlOx: a thin (~3 nm) skin on every exposed Al1 face, incl. the Al1/Al2
    # interface (so Al2 sits on top of the barrier) and the metal corners.
    _oxide_edges_cs(ax, al1, (solid2d == RESIST) | (solid2d == SUBSTRATE),
                    h_axis, zs)
    # The two arrows are the two electrode beams: evap 1 (Nb) and evap 3 (Al)
    # for a trilayer, evap 1 / evap 2 for a bilayer.
    c_e1 = _COLORS[C_E1_NB] if films2d is not None else _COLORS[C_AL1]
    c_e2 = _COLORS[C_E3_AL] if films2d is not None else _COLORS[C_AL2]
    lbl_e2 = "evap 3" if films2d is not None else "evap 2"
    _beam_arrow_cs(ax, r.meta["d1"], (ux, uy), half, vtop, c_e1,
                   "evap 1", side=-1, xcenter=view_center)
    _beam_arrow_cs(ax, r.meta["d2"], (ux, uy), half, vtop, c_e2,
                   lbl_e2, side=+1, xcenter=view_center)
    present = [c for c in sorted(np.unique(cat).tolist()) if c != C_EMPTY]
    _legend(fig, present)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def render_stages(r: DepositionResult, angle_deg=0.0, offset=0.0, junc_mask=None,
                  view_half=None, zmax=None, view_center=0.0, zmin=None):
    """Staged cross section sliced at in-plane azimuth ``angle_deg`` with
    perpendicular ``offset`` [nm].

    Bilayer → 5 panels: resist → evap1 → oxidation → evap2 → lift-off.
    Trilayer → 7 panels: resist → evap1 (Nb) → evap2 (Al) → oxidation →
    evap3 (Al) → evap4 (Nb) → lift-off."""
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    z_split = r.meta.get("z_split", r.z_top)
    solid2d, al1, al2, alox, h_axis, h_label, ix, iy, films2d = _slice_planes(
        r, angle_deg, offset)
    (ux, uy), _ = _slice_dirs(angle_deg)
    tri = films2d is not None
    # Bilayer evap-beam arrow colours; the trilayer branch below uses its own
    # per-evaporation colours (C_E1_NB … C_E4_NB) for each of the 4 arrows.
    c_e1 = _COLORS[C_AL1]
    c_e2 = _COLORS[C_AL2]

    if junc_mask is not None:
        jc = junc_mask[ix, iy]
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

    def _paint_metal(cat, inc_al1, inc_al2, mask=None):
        """Paint bilayer electrode metal (Al #1 / Al #2); ``mask`` (e.g.
        grounded) gates cells.  Trilayer panels use ``cat_tri`` (per-evap)."""
        def g(m):
            return m if mask is None else (m & mask)
        if inc_al1: cat[g(al1)] = C_AL1
        if inc_al2: cat[g(al2)] = C_AL2

    def cat_build(inc_al1=False, inc_al2=False,
                  liftoff=False, emphasise_junc=False):
        cat = np.full(solid2d.shape, C_EMPTY, np.int8)
        cat[sub] = C_SUBSTRATE
        if not liftoff:
            _resist_cat_cross(cat, solid2d, zs, z_split)
            _paint_metal(cat, inc_al1, inc_al2)
        else:
            # Lift-off: strip the resist; only grounded metal survives.
            _paint_metal(cat, True, True, mask=grounded)
            if emphasise_junc and junc2 is not None and not tri:
                # ~10 nm on EACH side of the AlOx barrier: top of grounded Al1
                # (below the oxide) + bottom of grounded Al2 (above it).  Keeps the
                # metal shape; recolours only the junction interface (not a
                # full-height rectangle).  ≥1 voxel/side; thinner where finer.
                dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
                n = max(1, int(round(10.0 / dz)))        # ≈10 nm per side, in voxels
                g1, g2 = al1 & grounded, al2 & grounded
                band = np.zeros_like(grounded)
                for i in np.where(jc & g1.any(1) & g2.any(1))[0]:
                    a1 = np.where(g1[i])[0]              # Al1 cells (lower electrode)
                    a2 = np.where(g2[i])[0]              # Al2 cells (upper electrode)
                    band[i, a1[-n:]] = True              # top ~10 nm of Al1 (below oxide)
                    band[i, a2[:n]] = True               # bottom ~10 nm of Al2 (above oxide)
                cat[band & (g1 | g2)] = C_JUNC
        return cat

    ox_excl_pre = (solid2d == RESIST) | sub
    half = _zoom_half(r, view_half)                      # scalar half-width (arrows)
    hwin = (view_center - half, view_center + half)      # panned (lo, hi) for _draw
    vlo = float(zs[0] if zmin is None else zmin)
    vhi = float(zs[-1] if zmax is None else zmax)
    vlim = (vlo, vhi)
    vtop = vhi

    # ── trilayer: 7-panel sequence (Evap1→Evap2→Ox→Evap3→Evap4→Lift-off) ──
    if tri:
        td = r.meta.get("tri_dirs", {})
        order = ["nb1", "al2", "al3", "nb4"]      # deposition order
        cols = {"nb1": C_E1_NB, "al2": C_E2_AL, "al3": C_E3_AL, "nb4": C_E4_NB}

        def cat_tri(inc, liftoff=False):
            cat = np.full(solid2d.shape, C_EMPTY, np.int8)
            cat[sub] = C_SUBSTRATE
            if liftoff:                            # resist stripped → grounded metal only
                for name in order:
                    cat[films2d[name] & grounded] = cols[name]
            else:
                _resist_cat_cross(cat, solid2d, zs, z_split)
                for name in order:                 # later films overwrite earlier
                    if name in inc:
                        cat[films2d[name]] = cols[name]
            return cat

        # (title, category grid, oxide phase, arrow=(film, colour, label, side))
        panels = [
            ("1. Resist only",   cat_tri([]),                           None,   None),
            ("2. Evap 1 (Nb)",   cat_tri(["nb1"]),                      None,   ("nb1", _COLORS[C_E1_NB], "evap 1", -1)),
            ("3. Evap 2 (Al)",   cat_tri(["nb1", "al2"]),               None,   ("al2", _COLORS[C_E2_AL], "evap 2", -1)),
            ("4. Oxidation",     cat_tri(["nb1", "al2"]),               "pre",  None),
            ("5. Evap 3 (Al)",   cat_tri(["nb1", "al2", "al3"]),        "pre",  ("al3", _COLORS[C_E3_AL], "evap 3", +1)),
            ("6. Evap 4 (Nb)",   cat_tri(["nb1", "al2", "al3", "nb4"]), "pre",  ("nb4", _COLORS[C_E4_NB], "evap 4", +1)),
            ("7. Lift-off (JJ)", cat_tri(order, liftoff=True),          "post", None),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.0), dpi=150,
                                 sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, (title, cat, ox, arrow) in zip(axes, panels):
            _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title,
                  hlim=hwin, vlim=vlim)
            if ox == "pre":       # resist still blocks oxidation
                _oxide_edges_cs(ax, al1, ox_excl_pre, h_axis, zs)
            elif ox == "post":    # lift-off: only grounded electrode-1 oxidises
                _oxide_edges_cs(ax, al1 & grounded, sub, h_axis, zs)
            if arrow is not None:
                name, col, lbl, side = arrow
                d = td.get(name, r.meta["d1"])
                _beam_arrow_cs(ax, d, (ux, uy), half, vtop, col, lbl, side=side,
                               xcenter=view_center)
        for ax in axes[len(panels):]:
            ax.axis("off")       # unused cells (7 stages in a 2×4 grid)
        _legend(fig, [C_SUBSTRATE, C_RESIST_LO, C_RESIST_UP,
                      C_E1_NB, C_E2_AL, C_E3_AL, C_E4_NB])
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        return fig

    # ── bilayer: 5-panel sequence (Evap1→Ox→Evap2→Lift-off) ──
    # oxide appears once Al1 is present and stays through evap-2 / lift-off.
    # Drawn as a thin skin on the exposed Al1 faces: before lift-off the resist
    # (and substrate) block oxidation; at lift-off the resist is gone so only
    # the surviving (grounded) Al1 oxidises against air / Al2.
    panels = [
        ("1. Resist only",   cat_build(),                                False),
        ("2. Evaporation 1", cat_build(inc_al1=True),                    False),
        ("3. Oxidation",     cat_build(inc_al1=True),                    True),
        ("4. Evaporation 2", cat_build(inc_al1=True, inc_al2=True),      True),
        ("5. Lift-off (JJ)", cat_build(liftoff=True, emphasise_junc=True), True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), dpi=150,
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for k, (ax, (title, cat, show_ox)) in enumerate(zip(axes, panels)):
        _draw(ax, cat, h_axis, zs, h_label, "z  [nm]", title, hlim=hwin, vlim=vlim)
        if show_ox:
            if k == 4:        # lift-off: resist stripped, grounded Al1 only
                _oxide_edges_cs(ax, al1 & grounded, sub, h_axis, zs)
            else:
                _oxide_edges_cs(ax, al1, ox_excl_pre, h_axis, zs)
        if k in (1, 2):       # evap-1 related panels
            _beam_arrow_cs(ax, r.meta["d1"], (ux, uy), half, vtop,
                           c_e1, "evap 1", side=-1, xcenter=view_center)
        if k == 3:            # evap-2 panel
            _beam_arrow_cs(ax, r.meta["d2"], (ux, uy), half, vtop,
                           c_e2, "evap 2", side=+1, xcenter=view_center)
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


def _film_floor_maps(r):
    """Per-film floor footprints {nb1f, al2f, al3f, nb4f} as (Nx,Ny) bools for a
    trilayer result, else ``None`` (bilayer)."""
    if getattr(r, "stack", "Bilayer") != "Trilayer" or not r.films:
        return None
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    floor = (zs >= 0) & (zs < zf)
    return {k: g[:, :, floor].any(axis=2) for k, g in r.films.items()}


def _resist_cat_top(cat, upper_resist, lower_resist):
    """Paint the resist opening pattern (incl. undercut shelf) from above."""
    # solid resist (both layers) = upper imaging colour
    cat[upper_resist & lower_resist] = C_RESIST_UP
    # open below but resist above = the undercut shelf / overhang
    cat[upper_resist & ~lower_resist] = C_RESIST_LO
    # through-hole (open both) stays empty (substrate visible)


def render_top_stages(r: DepositionResult, junc_mask=None, view_half=None,
                      juncs=None, combo_map=None):
    """Staged top view (resist with undercut → … → lift-off).

    The resist / undercut pattern is drawn opaque underneath, then the deposited
    metal is layered on top semi-transparently so the resist mask stays visible.
    The AlOx barrier is filled (as squares) on top of the Al1 footprint, and any
    separate Josephson junctions are labelled J1, J2, … in the lift-off panel.
    For a trilayer, pass ``combo_map`` to colour the junction by Nb-Al / Al-Al /
    Nb-Nb in the lift-off panel.  A trilayer expands to 7 panels:
    resist → evap1 (Nb) → evap2 (Al) → oxidation → evap3 (Al) → evap4 (Nb) →
    lift-off.
    """
    al1f, al2f, aloxf, upper_resist, lower_resist = _floor_maps(r)
    films2d = _film_floor_maps(r)

    # Opaque resist base (substrate visible through the open holes).
    resist_base = np.full(al1f.shape, C_EMPTY, np.int8)
    _resist_cat_top(resist_base, upper_resist, lower_resist)
    empty_base = np.full(al1f.shape, C_EMPTY, np.int8)

    # ── trilayer: 7-panel top-view sequence ──
    if films2d is not None:
        td = r.meta.get("tri_dirs", {})
        order = ["nb1", "al2", "al3", "nb4"]
        cols = {"nb1": C_E1_NB, "al2": C_E2_AL, "al3": C_E3_AL, "nb4": C_E4_NB}
        elec2f = films2d["al3"] | films2d["nb4"]      # electrode-2 floor footprint

        def metal_tri(inc, emphasise_junc=False):
            cat = np.full(al1f.shape, C_EMPTY, np.int8)
            for name in order:                         # later films overwrite earlier
                if name in inc:
                    cat[films2d[name]] = cols[name]
            if emphasise_junc:
                _paint_junction_top(cat, junc_mask, combo_map)
            return cat

        ox_post = aloxf & ~elec2f          # electrode-2 sits over the barrier
        # (title, base, metal grid, alpha, oxide mask, label junctions, arrow)
        panels = [
            ("1. Resist only", resist_base, metal_tri([]),                            0.62, None,    False, None),
            ("2. Evap 1 (Nb)", resist_base, metal_tri(["nb1"]),                       0.62, None,    False, ("nb1", C_E1_NB, "evap 1")),
            ("3. Evap 2 (Al)", resist_base, metal_tri(["nb1", "al2"]),                0.62, None,    False, ("al2", C_E2_AL, "evap 2")),
            ("4. Oxidation",   resist_base, metal_tri(["nb1", "al2"]),                0.62, aloxf,   False, None),
            ("5. Evap 3 (Al)", resist_base, metal_tri(["nb1", "al2", "al3"]),         0.62, ox_post, False, ("al3", C_E3_AL, "evap 3")),
            ("6. Evap 4 (Nb)", resist_base, metal_tri(["nb1", "al2", "al3", "nb4"]),  0.62, ox_post, False, ("nb4", C_E4_NB, "evap 4")),
            ("7. Lift-off (JJ)", empty_base,
             metal_tri(order, emphasise_junc=True),                                  1.00, ox_post, True,  None),
        ]
        hw = _top_half(r, view_half)
        fig, axes = plt.subplots(2, 4, figsize=(17.5, 9.0),
                                 sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, (title, base, mcat, malpha, ox_mask, lbl, arrow) in zip(axes, panels):
            _draw(ax, base, r.xs, r.ys, "x  [nm]", "y  [nm]", title)
            _overlay_cats(ax, mcat, r.xs, r.ys, _TRI_METAL_CATS,
                          alpha=malpha, zorder=3)
            if ox_mask is not None:
                _oxide_fill(ax, ox_mask, r.xs, r.ys, alpha=0.5, zorder=4)
            ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
            ax.set_aspect("equal")
            if lbl:
                _junc_labels(ax, juncs)
            if arrow is not None:
                name, col, albl = arrow
                d = td.get(name, r.meta["d1"])
                _beam_arrow_top(ax, d, hw, _COLORS[col], albl)
        for ax in axes[len(panels):]:
            ax.axis("off")
        _legend(fig, [C_RESIST_LO, C_RESIST_UP, C_E1_NB, C_E2_AL, C_E3_AL, C_E4_NB,
                      C_JUNC_NBAL, C_JUNC_ALAL, C_JUNC_NBNB])
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        return fig

    # ── bilayer: 5-panel top-view sequence ──
    def metal_cat(inc_al1=False, inc_al2=False, emphasise_junc=False):
        cat = np.full(al1f.shape, C_EMPTY, np.int8)
        if inc_al1: cat[al1f] = C_AL1
        if inc_al2: cat[al2f] = C_AL2
        if emphasise_junc:
            _paint_junction_top(cat, junc_mask, combo_map)
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
        _overlay_cats(ax, mcat, r.xs, r.ys, _METAL_CATS, alpha=malpha, zorder=3)
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
    if combo_map is not None:
        _legend(fig, [C_RESIST_LO, C_RESIST_UP, C_AL1, C_AL2,
                      C_JUNC_NBAL, C_JUNC_ALAL, C_JUNC_NBNB])
    else:
        _legend(fig, [C_RESIST_LO, C_RESIST_UP, C_AL1, C_AL2, C_JUNC])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def _slice_marker(ax, slice_line, hw):
    """Draw a dashed line on the top view marking the cross-section slice.

    ``slice_line`` is ``(angle_deg, offset)``: the slice runs along the in-plane
    direction u(α) and is shifted by ``offset`` along the perpendicular n(α).
    """
    if slice_line is None:
        return
    angle_deg, offset = slice_line
    (ux, uy), (nx, ny) = _slice_dirs(angle_deg)
    col = "#ffd166"
    cx, cy = offset * nx, offset * ny           # a point on the slice line
    L = 3.0 * hw                                # overshoot; axes clip it
    x0, y0 = cx - L * ux, cy - L * uy
    x1, y1 = cx + L * ux, cy + L * uy
    ax.plot([x0, x1], [y0, y1], color=col, lw=1.8, ls="--", zorder=9)
    # label near the +u end, kept inside the view
    lx, ly = cx + 0.82 * hw * ux, cy + 0.82 * hw * uy
    rot = ((angle_deg + 90) % 180) - 90         # keep text upright
    ax.text(lx, ly, f"α={angle_deg:.0f}°, d={offset:.0f}", color=col,
            fontsize=8, ha="center", va="bottom", rotation=rot,
            rotation_mode="anchor", fontweight="bold", zorder=10)


_METAL_CATS = [C_AL1, C_AL2, C_JUNC, C_JUNC_NBAL, C_JUNC_ALAL, C_JUNC_NBNB]
# trilayer top-view metals (per evaporation) + junction-combo overlays
_TRI_METAL_CATS = [C_E1_NB, C_E2_AL, C_E3_AL, C_E4_NB,
                   C_JUNC_NBAL, C_JUNC_ALAL, C_JUNC_NBNB]


def _paint_junction_top(mcat, junc_mask, combo_map):
    """Colour the junction footprint: by Nb-Al/Al-Al/Nb-Nb combo for a trilayer
    (``combo_map`` given), else a single junction colour."""
    if combo_map is not None:
        for code, cat in _COMBO_CAT.items():
            mcat[combo_map == code] = cat
    elif junc_mask is not None:
        mcat[junc_mask] = C_JUNC


def render_top_view(r: DepositionResult, junc_mask=None, view_half=None,
                    juncs=None, slice_line=None, combo_map=None):
    """Single top-down floor map: resist/undercut (opaque) with Al1 / Al2 /
    junction layered on top semi-transparently so the resist mask stays visible.
    The AlOx barrier is filled on top of Al1 and separate junctions are labelled.
    When ``slice_line=(angle_deg, offset)`` is given, the (possibly rotated)
    cross-section cut is marked with a dashed line.  For a trilayer, pass
    ``combo_map`` to colour the junction by Nb-Al / Al-Al / Nb-Nb."""
    al1f, al2f, aloxf, upper_resist, lower_resist = _floor_maps(r)
    base = np.full(al1f.shape, C_EMPTY, np.int8)
    _resist_cat_top(base, upper_resist, lower_resist)

    mcat = np.full(al1f.shape, C_EMPTY, np.int8)
    mcat[al1f] = C_AL1
    mcat[al2f] = C_AL2
    _paint_junction_top(mcat, junc_mask, combo_map)

    hw = _top_half(r, view_half)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    _draw(ax, base, r.xs, r.ys, "x  [nm]", "y  [nm]", "Top view (floor deposit)")
    _overlay_cats(ax, mcat, r.xs, r.ys, _METAL_CATS, alpha=0.62, zorder=3)
    _oxide_fill(ax, aloxf & ~al2f, r.xs, r.ys, alpha=0.5, zorder=4)
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
    ax.set_aspect("equal")
    _junc_labels(ax, juncs)
    _slice_marker(ax, slice_line, hw)
    _beam_arrow_top(ax, r.meta["d1"], hw, _COLORS[C_AL1], "evap 1")
    _beam_arrow_top(ax, r.meta["d2"], hw, _COLORS[C_AL2], "evap 2")
    present = [c for c in (C_RESIST_LO, C_RESIST_UP) if (base == c).any()]
    present += [c for c in (C_AL1, C_AL2) + tuple(_COMBO_CAT.values()) + (C_JUNC,)
                if (mcat == c).any()]
    if present:
        _legend(fig, present)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ════════════════════════════════════════════════════════════════
# lift-off film-thickness maps
# ════════════════════════════════════════════════════════════════

def _thickness_field(r: DepositionResult):
    """Per-(x,y) lift-off metal-film thickness [nm] as an (Nx,Ny) float grid.

    After lift-off only the in-trench metal survives (the same band used by
    ``_floor_maps`` / ``_junction_cells_3d``: 0 ≤ z < z_floor).  At each column
    the thickness is the count of stacked metal voxels (either electrode) times
    the voxel edge, so where the two electrodes overlap — the junction — the
    stack is thicker, and single-electrode regions are thinner."""
    zs = r.zs
    zf = r.meta.get("z_floor", r.z_top)
    floor = (zs >= 0) & (zs < zf)
    metal = (r.al1 | r.al2)[:, :, floor]
    return metal.sum(axis=2).astype(float) * r.vox          # (Nx, Ny) [nm]


def render_thickness_map(r: DepositionResult, view_half=None):
    """Top-down heat map of the lift-off metal-film thickness.

    Colour encodes the stacked-metal thickness [nm] at each (x, y); the
    electrode overlap (junction) reads as a thicker ridge.  Cells with no metal
    are left blank (background)."""
    thick = _thickness_field(r)
    masked = np.ma.masked_where(thick <= 0, thick)

    hw = _top_half(r, view_half)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(plt.rcParams["axes.facecolor"])            # no-metal = background
    im = ax.imshow(masked.T, origin="lower", extent=_extent(r.xs, r.ys),
                   aspect="equal", cmap=cmap, interpolation="nearest", zorder=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("metal thickness  [nm]")
    ax.set_xlabel("x  [nm]")
    ax.set_ylabel("y  [nm]")
    ax.set_title("Lift-off metal thickness (heat map)")
    ax.set_xlim(-hw, hw)
    ax.set_ylim(-hw, hw)
    fig.tight_layout()
    return fig


def render_thickness_surface(r: DepositionResult, view_half=None):
    """3-D surface of the lift-off metal-film thickness (height = thickness).

    Same thickness field as :func:`render_thickness_map`, drawn as a surface
    z = thickness(x, y) over the view window.  No-metal regions sit at z = 0
    (substrate level)."""
    thick = _thickness_field(r)

    hw = _top_half(r, view_half)
    xsel = np.abs(r.xs) <= hw
    ysel = np.abs(r.ys) <= hw
    if not xsel.any():
        xsel[:] = True                                      # degenerate-window guard
    if not ysel.any():
        ysel[:] = True
    X, Y = np.meshgrid(r.xs[xsel], r.ys[ysel], indexing="ij")
    Z = thick[np.ix_(xsel, ysel)]

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.08, label="thickness  [nm]")
    ax.set_xlabel("x  [nm]")
    ax.set_ylabel("y  [nm]")
    ax.set_zlabel("thickness  [nm]")
    ax.set_title("Lift-off metal thickness (3D)")
    # dark-theme the 3D panes (rcParams styling does not reach the 3D panes)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.05, 0.07, 0.09, 1.0))
    fig.tight_layout()
    return fig


def render_thickness_surface_plotly(r: DepositionResult, view_half=None):
    """Interactive (drag-rotate / zoom / pan) Plotly 3-D surface of the lift-off
    metal-film thickness — same field/window as :func:`render_thickness_surface`.

    Returns a ``plotly.graph_objects.Figure``; ``plotly`` is imported lazily so
    this module still imports if plotly is absent (the caller falls back to the
    matplotlib surface)."""
    import plotly.graph_objects as go

    thick = _thickness_field(r)
    hw = _top_half(r, view_half)
    xsel = np.abs(r.xs) <= hw
    ysel = np.abs(r.ys) <= hw
    if not xsel.any():
        xsel[:] = True                                      # degenerate-window guard
    if not ysel.any():
        ysel[:] = True
    Z = thick[np.ix_(xsel, ysel)]                           # (Nx, Ny)
    # go.Surface expects z shaped (len(y), len(x)) → transpose.
    fig = go.Figure(go.Surface(
        x=r.xs[xsel], y=r.ys[ysel], z=Z.T, colorscale="Viridis",
        colorbar=dict(title="thickness [nm]")))
    fig.update_layout(
        template="plotly_dark", title="Lift-off metal thickness (3D)",
        scene=dict(xaxis_title="x [nm]", yaxis_title="y [nm]",
                   zaxis_title="thickness [nm]", aspectmode="auto"),
        margin=dict(l=0, r=0, t=30, b=0), height=520,
        paper_bgcolor="#0e1117", font=dict(color="#e6e6e6"))
    return fig


# ── Plassys source / tilted-wafer schematic ────────────────────────
def render_wafer_geometry(p, L, flat_ratio=0.65):
    """Per-evaporation schematic: fixed point source BELOW + tilted wafer, each
    evaporation shown as a compact 2×2 block — 3-D perspective + orthographic
    top / front / side projections grouped around it.

    One 2×2 block per active evaporation (2 bilayer/Manhattan, 4 trilayer),
    tiled in a near-square meta-grid.  Within a block: 3-D (top-left), top view
    x–y (top-right), front view x–z (bottom-left), side view y–z (bottom-right).
    Real-Plassys layout: the fixed source sits at the origin (red dot, below) and
    the beam goes UP to the wafer (disk + primary flat オリフラ) held centre-up at
    z = +L and tilted by R = Ry(θ)·Rz(−φ).  Orange arrow = wafer normal (toward
    the source); dashed red = vertical beam source→centre.  ``flat_ratio`` =
    flat chord / wafer radius (matches the selected wafer; 0.65 ≈ 4-inch).
    """
    beams = pe.evap_beams(p)
    n = len(beams)
    mcols = int(np.ceil(np.sqrt(n)))             # evap blocks in a near-square grid
    mrows = int(np.ceil(n / mcols))
    fig = plt.figure(figsize=(6.8 * mcols, 6.4 * mrows))
    gs = fig.add_gridspec(2 * mrows, 2 * mcols)
    pane = (0.05, 0.07, 0.09, 1.0)
    flip = np.array([1.0, 1.0, -1.0])           # source below / beam upward
    C = np.array([0.0, 0.0, float(L)])          # wafer centre ABOVE the source
    s = 0.4 * float(L)                           # drawing wafer radius
    chord = float(flat_ratio) * s
    d = float(np.sqrt(max(s * s - (chord / 2.0) ** 2, 0.0)))
    uv = _wafer_path(s, chord, d).vertices       # (M,2) wafer outline, in-plane

    def _proj(ax, pts, nrm, i, j, xl, yl, ttl):
        ax.plot(pts[:, i], pts[:, j], color="#64B5F6", lw=1.6)
        ax.fill(pts[:, i], pts[:, j], color="#64B5F6", alpha=0.20, lw=0)
        ax.annotate("", xy=(C[i] + 0.5 * s * nrm[i], C[j] + 0.5 * s * nrm[j]),
                    xytext=(C[i], C[j]),
                    arrowprops=dict(arrowstyle="-|>", color="#FFB74D", lw=2))
        ax.plot([0, C[i]], [0, C[j]], color="#EF9A9A", ls="--", lw=1.4)
        ax.scatter([0], [0], color="#EF9A9A", s=40, zorder=5)
        ax.set_aspect("equal"); ax.margins(0.12); ax.set_facecolor(pane)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_title(ttl, fontsize=8); ax.tick_params(labelsize=7)

    for kk, (lbl, _ta, _pa, th, ph) in enumerate(beams):
        br, bc = divmod(kk, mcols)
        r0, c0 = 2 * br, 2 * bc
        Rm = pe._wafer_rot(th, ph)
        eX, eY, nrm = Rm[:, 0] * flip, Rm[:, 1] * flip, Rm[:, 2] * flip
        pts = C + uv[:, 0:1] * eX + uv[:, 1:2] * eY     # tilted wafer rim (M,3)

        ax = fig.add_subplot(gs[r0, c0], projection="3d")     # block top-left — 3-D
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#64B5F6", lw=1.8)
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2],
                        color="#64B5F6", alpha=0.22, linewidth=0)
        ax.quiver(C[0], C[1], C[2], 0.5 * s * nrm[0], 0.5 * s * nrm[1],
                  0.5 * s * nrm[2], color="#FFB74D", lw=2)
        ax.plot([0, C[0]], [0, C[1]], [0, C[2]],
                color="#EF9A9A", ls="--", lw=1.5)
        ax.scatter([0], [0], [0], color="#EF9A9A", s=45)
        ax.set_title(f"{lbl}\nθ={th:.0f}°  φ={ph:.0f}°", fontsize=9)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color(pane)
        ax.set_xlabel("x  [mm]", fontsize=8); ax.set_ylabel("y  [mm]", fontsize=8)
        ax.set_zlabel("z  [mm]", fontsize=8)

        _proj(fig.add_subplot(gs[r0, c0 + 1]), pts, nrm, 0, 1,     # top-right
              "x  [mm]", "y  [mm]", "top (x–y)")
        _proj(fig.add_subplot(gs[r0 + 1, c0]), pts, nrm, 0, 2,     # bottom-left
              "x  [mm]", "z  [mm]", "front (x–z)")
        _proj(fig.add_subplot(gs[r0 + 1, c0 + 1]), pts, nrm, 1, 2, # bottom-right
              "y  [mm]", "z  [mm]", "side (y–z)")

    fig.suptitle("Fixed point source below (red) + tilted wafer (with flat): "
                 "3-D + orthographic top / front / side projections", fontsize=11)
    fig.tight_layout()
    return fig


def _wafer_path(R, c, d, npts=200):
    """matplotlib ``Path`` of the wafer boundary: the circular arc plus the
    bottom primary flat (オリフラ).

    ``R`` = wafer radius, ``c`` = primary-flat chord length, ``d`` = distance of
    the flat chord from the centre (= sqrt(R² − (c/2)²)); the flat sits at y = −d.
    """
    a0 = np.arctan2(-d,  c / 2.0)                 # right flat endpoint
    a1 = np.arctan2(-d, -c / 2.0) + 2 * np.pi     # left endpoint, the long way (top)
    t = np.linspace(a0, a1, npts)
    verts = np.column_stack([R * np.cos(t), R * np.sin(t)]).tolist()
    verts += [[-c / 2.0, -d], [c / 2.0, -d]]      # close across the flat
    return Path(verts)


def render_wafer_map_2d(coords, area, ic, R, c, d, title=""):
    """Draw the wafer-position maps on an actual wafer disk + primary flat.

    Two side-by-side panels (junction area, est. Ic), each a ``pcolormesh`` over
    ``meshgrid(coords, coords)`` clipped to the wafer outline so it reads as a
    real wafer; off-wafer (NaN) cells are masked out, the wafer boundary + flat
    are drawn, grid-cell centres are dotted, and the centre cross marks the
    nominal (single-JJ) position.
    """
    fig, (axa, axi) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    Xg, Yg = np.meshgrid(coords, coords)
    for ax, Z, cmap, lbl in ((axa, area, "viridis", "Junction area  [nm²]"),
                             (axi, ic,   "magma",   "Est. Ic  [µA]")):
        pcm = ax.pcolormesh(coords, coords, np.ma.masked_invalid(Z),
                            shading="auto", cmap=cmap)
        patch = PathPatch(_wafer_path(R, c, d), transform=ax.transData,
                          fc="none", ec="none")
        ax.add_patch(patch); pcm.set_clip_path(patch)           # confine fill to wafer
        ax.add_patch(PathPatch(_wafer_path(R, c, d), transform=ax.transData,
                               fc="none", ec="#90A4AE", lw=1.6))  # wafer outline + flat
        ax.scatter(Xg, Yg, s=5, c="#455A64", alpha=0.5)         # grid-cell dots
        ax.plot(0, 0, "+", color="w", ms=12, mew=1.8)           # centre = nominal
        fig.colorbar(pcm, ax=ax, label=lbl)
        ax.set_xlabel("wafer x  [mm]"); ax.set_ylabel("wafer y  [mm]")
        ax.set_aspect("equal")
        ax.set_xlim(-R * 1.05, R * 1.05); ax.set_ylim(-R * 1.05, R * 1.05)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
