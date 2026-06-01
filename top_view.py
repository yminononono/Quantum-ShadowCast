"""
top_view.py  (v6)
=================
Correct top-view renderer.

Dolan bridge:
  Deposits CLIPPED to MMA gap (x ∈ [-mma_hx, +mma_hx], y ∈ [-mma_hy, +mma_hy]).
  Evap1 (θ<0, beam from RIGHT): deposit on RIGHT side of bridge.
  Evap2 (θ>0, beam from LEFT ): deposit on LEFT  side of bridge.
  Junction = center overlap under bridge.

Manhattan cross:
  Deposits clipped to cross-shaped PMMA+MMA opening.
  Each evaporation forms a cross-shaped deposit.
  φ₁ and φ₂ are independent.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.transforms as transforms
from process_engine import ProcessParams, shadow_vector
from junction_area import compute_junction_area

C_PMMA    = "#FFB300"
C_MMA_UC  = "#FFF59D"
C_BRIDGE  = "#66BB6A"
C_SHADOW1 = "#64B5F6"
C_SHADOW2 = "#EF9A9A"
C_JJ      = "#CE93D8"
C_BG      = "#1A1A2E"


def draw_top_view(polygons, params: ProcessParams,
                  show_shadow=True, show_undercut=True,
                  mode=None) -> plt.Figure:
    mode = mode or params.mode
    if mode == "Dolan bridge":
        return _dolan_top_view(params, show_shadow, show_undercut)
    else:
        return _manhattan_top_view(params, show_shadow, show_undercut)


# ─── Dolan bridge ────────────────────────────────────────────────

def _dolan_top_view(p: ProcessParams, show_shadow, show_undercut) -> plt.Figure:
    res    = compute_junction_area(p)
    L      = p.bridge_len
    W      = p.bridge_w
    u      = p.undercut
    mma_hx = L/2 + u
    mma_hy = W/2 + u
    pad_x  = 1400   # electrode pad half-length in x
    pad_y  = max(mma_hy + 400, 700)   # electrode pad half-height

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor(C_BG)
    ax.set_aspect("equal")

    # ── MMA air gap (visible through bridge from above) ───────────
    # Full MMA opening: x ∈ [-mma_hx, +mma_hx], y ∈ [-mma_hy, +mma_hy]
    if show_undercut:
        ax.add_patch(Rectangle(
            (-mma_hx, -mma_hy), 2*mma_hx, 2*mma_hy,
            fc=C_MMA_UC, ec="#F57F17", lw=1.2, ls="--",
            alpha=0.35, zorder=2, label="MMA undercut gap"))

    # ── PMMA structure ────────────────────────────────────────────
    # Left electrode: x ∈ [-mma_hx-pad_x, -mma_hx], y ∈ [-pad_y, +pad_y]
    ax.add_patch(Rectangle(
        (-mma_hx-pad_x, -pad_y), pad_x, 2*pad_y,
        fc=C_PMMA, ec="#E65100", lw=1.2, alpha=0.85, zorder=3,
        label="PMMA electrode"))
    # Right electrode
    ax.add_patch(Rectangle(
        (mma_hx, -pad_y), pad_x, 2*pad_y,
        fc=C_PMMA, ec="#E65100", lw=1.2, alpha=0.85, zorder=3))

    # PMMA walls above and below bridge (y ∈ [+W/2, +mma_hy+...] etc.)
    # These walls span x ∈ [-mma_hx, +mma_hx] but only outside bridge width
    for sign in [-1, 1]:
        y_inner = sign * W/2        # bridge edge in y
        y_outer = sign * pad_y      # outer extent
        h = y_outer - y_inner
        if abs(h) > 0:
            ax.add_patch(Rectangle(
                (-mma_hx-pad_x, min(y_inner, y_outer)),
                2*mma_hx + 2*pad_x, abs(h),
                fc=C_PMMA, ec="#E65100", lw=0.6, alpha=0.85, zorder=3))

    # ── Shadow deposits (clipped to MMA gap) ──────────────────────
    if show_shadow:
        ev1_xlo = res["ev1_x_lo"];  ev1_xhi = res["ev1_x_hi"]
        ev2_xlo = res["ev2_x_lo"];  ev2_xhi = res["ev2_x_hi"]
        y1_lo   = res["y1_lo"];     y1_hi   = res["y1_hi"]
        y2_lo   = res["y2_lo"];     y2_hi   = res["y2_hi"]

        # Clip to MMA gap bounds
        ev1_xlo = max(ev1_xlo, -mma_hx);  ev1_xhi = min(ev1_xhi, mma_hx)
        ev2_xlo = max(ev2_xlo, -mma_hx);  ev2_xhi = min(ev2_xhi, mma_hx)
        y1_lo   = max(y1_lo, -mma_hy);    y1_hi   = min(y1_hi, mma_hy)
        y2_lo   = max(y2_lo, -mma_hy);    y2_hi   = min(y2_hi, mma_hy)

        # Evap1 deposit (RIGHT side, θ<0)
        if ev1_xlo < ev1_xhi and y1_lo < y1_hi:
            ax.add_patch(Rectangle(
                (ev1_xlo, y1_lo), ev1_xhi-ev1_xlo, y1_hi-y1_lo,
                fc=C_SHADOW1, ec="#1565C0", lw=0.8, alpha=0.55, zorder=4,
                label=f"Al₁  θ={p.angle1}° φ={p.phi1}°  (from right)"))

        # Evap2 deposit (LEFT side, θ>0)
        if ev2_xlo < ev2_xhi and y2_lo < y2_hi:
            ax.add_patch(Rectangle(
                (ev2_xlo, y2_lo), ev2_xhi-ev2_xlo, y2_hi-y2_lo,
                fc=C_SHADOW2, ec="#B71C1C", lw=0.8, alpha=0.55, zorder=4,
                label=f"Al₂  θ={p.angle2}° φ={p.phi2}°  (from left)"))

        # Junction overlap
        jrect = res["junction_rect"]
        if jrect:
            jxs = [c[0] for c in jrect] + [jrect[0][0]]
            jys = [c[1] for c in jrect] + [jrect[0][1]]
            ax.fill(jxs, jys, color=C_JJ, alpha=0.90, zorder=6,
                    label=f"Junction  A={res['area_nm2']:.0f} nm²")
            ax.plot(jxs, jys, color="white", lw=1.8, zorder=7)

    # ── Bridge slab (on top, z-order highest among resist) ────────
    ax.add_patch(Rectangle(
        (-L/2, -W/2), L, W,
        fc=C_BRIDGE, ec="#1B5E20", lw=2.0, alpha=0.92, zorder=5,
        label=f"Bridge slab  {L:.0f}×{W:.0f} nm"))
    ax.text(0, 0, f"bridge\n{L:.0f}nm × {W:.0f}nm",
            ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")

    # ── Evaporation direction arrows ──────────────────────────────
    for angle, phi, color, side in [
        (p.angle1, p.phi1, C_SHADOW1, "→ from right"),
        (p.angle2, p.phi2, C_SHADOW2, "← from left"),
    ]:
        # Projected direction in x-y plane
        phi_rad = np.radians(phi)
        dx_u =  np.cos(phi_rad)   # unit x
        dy_u =  np.sin(phi_rad)   # unit y
        arr = 400
        x0 = 0;  y0 = mma_hy + 420 + (60 if angle > 0 else 0)
        ax.annotate("", xy=(x0 + arr*dx_u, y0 + arr*dy_u),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2))
        ax.text(x0, y0 - 35,
                f"θ={angle}° φ={phi}° {side}",
                ha="center", va="top", fontsize=8,
                color=color, fontweight="bold")

    # ── Dimension annotations ─────────────────────────────────────
    y_ann = -mma_hy - 90
    ax.annotate("", xy=(-L/2, y_ann), xytext=(L/2, y_ann),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.2))
    ax.text(0, y_ann-18, f"bridge_len = {L:.0f} nm",
            ha="center", va="top", fontsize=8, color="white")
    x_ann = mma_hx + 80
    ax.annotate("", xy=(x_ann, -W/2), xytext=(x_ann, W/2),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.2))
    ax.text(x_ann+15, 0, f"bridge_w\n{W:.0f} nm",
            ha="left", va="center", fontsize=8, color="white")
    # undercut arrow
    ax.annotate("", xy=(L/2, W/2), xytext=(mma_hx, mma_hy),
                arrowprops=dict(arrowstyle="<->", color="#FFB300", lw=1.0))
    ax.text(mma_hx+15, mma_hy+15, f"u={u:.0f}nm",
            fontsize=7.5, color="#FFB300")

    # ── Info box ──────────────────────────────────────────────────
    r = res
    t_sh = p.t_mma*np.tan(np.radians(abs(p.angle1)))
    info = (f"overlap_x = {r['overlap_x_nm']:.1f} nm\n"
            f"overlap_y = {r['overlap_y_nm']:.1f} nm\n"
            f"Area = {r['area_nm2']:.0f} nm²\n"
            f"Ic ≈ {r['ic_estimate_uA']:.3f} µA\n"
            f"shadow = {t_sh:.0f} nm\n"
            f"{'OK: junction' if r['area_nm2']>0 else 'OPEN (need bridge_len < '+str(int(2*t_sh))+' nm)'}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top",
            fontsize=8.5, color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#2D2D44", ec=C_JJ, lw=1.2))

    # ── Axes ─────────────────────────────────────────────────────
    xlim_w = mma_hx + pad_x + 200
    ax.set_xlim(-xlim_w, xlim_w)
    ax.set_ylim(-pad_y - 120, pad_y + 700)
    ax.set_xlabel("x [nm]  (evaporation direction)", color="white", fontsize=10)
    ax.set_ylabel("y [nm]  (bridge width direction)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#555")
    ax.grid(True, color="#333", alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8, facecolor="#2D2D44",
              labelcolor="white", framealpha=0.9)
    ax.set_title(
        f"Dolan bridge — top view  "
        f"bridge_len={L:.0f} nm  bridge_w={W:.0f} nm  undercut={u:.0f} nm",
        color="white", fontsize=11, fontweight="bold")

    fig.patch.set_facecolor(C_BG)
    fig.tight_layout()
    return fig


# ─── Manhattan cross ─────────────────────────────────────────────

def _manhattan_top_view(p: ProcessParams, show_shadow, show_undercut) -> plt.Figure:
    """Double-oblique Manhattan top view (arxiv:2605.19590).

    Two perpendicular resist line openings (designed widths wx, wy) cross at
    the origin. Double-oblique shadowing narrows each deposited line to
    w_narrow = w_open − h·sinδ/tanθ. Junction = crossing of narrowed lines.
    """
    res    = compute_junction_area(p)
    wx     = p.manhattan_wx          # x-running arm opening width (y-extent)
    wy     = p.manhattan_wy          # y-running arm opening width (x-extent)
    wnx    = res["wnarrow_x_nm"]     # narrowed x-arm width (y-extent)
    wny    = res["wnarrow_y_nm"]     # narrowed y-arm width (x-extent)
    delta  = res["delta_deg"]
    theta  = res["theta_deg"]
    arm_len = 2500

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(C_BG)
    ax.set_aspect("equal")

    # ── Designed resist openings (two crossing lines) ─────────────
    # x-running arm: long in x, width wx in y.
    ax.add_patch(Rectangle((-arm_len, -wx/2), 2*arm_len, wx,
                 fc=C_PMMA, ec="#E65100", lw=0.8, alpha=0.30, zorder=2,
                 label=f"Resist opening (w_open={wx:.0f}/{wy:.0f} nm)"))
    # y-running arm: long in y, width wy in x.
    ax.add_patch(Rectangle((-wy/2, -arm_len), wy, 2*arm_len,
                 fc=C_PMMA, ec="#E65100", lw=0.8, alpha=0.30, zorder=2))

    # ── Narrowed deposited lines ──────────────────────────────────
    if show_shadow:
        # Evap1 → x-running arm, narrowed to wnx in y
        if wnx > 0:
            ax.add_patch(Rectangle((-arm_len, -wnx/2), 2*arm_len, wnx,
                         fc=C_SHADOW1, ec="#1565C0", lw=0.8, alpha=0.55, zorder=4,
                         label=f"Al₁ line  w={wnx:.0f} nm"))
        # Evap2 → y-running arm, narrowed to wny in x
        if wny > 0:
            ax.add_patch(Rectangle((-wny/2, -arm_len), wny, 2*arm_len,
                         fc=C_SHADOW2, ec="#B71C1C", lw=0.8, alpha=0.55, zorder=4,
                         label=f"Al₂ line  w={wny:.0f} nm"))
        # Junction = crossing
        jrect = res["junction_rect"]
        if jrect:
            jxs = [c[0] for c in jrect]+[jrect[0][0]]
            jys = [c[1] for c in jrect]+[jrect[0][1]]
            ax.fill(jxs, jys, color=C_JJ, alpha=0.95, zorder=6,
                    label=f"Junction  A={res['area_nm2']:.0f} nm²")
            ax.plot(jxs, jys, color="white", lw=1.8, zorder=7)

    # ── Double-oblique evaporation arrows (offset δ from each arm) ─
    L = wx/2 + 700
    # Evap1: beam azimuth offset δ from the x-arm direction
    a1 = np.radians(delta)
    ax.annotate("", xy=(550*np.cos(a1), 550*np.sin(a1)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_SHADOW1, lw=2.4), zorder=8)
    # Evap2: beam azimuth offset δ from the y-arm direction (90°)
    a2 = np.radians(90 - delta)
    ax.annotate("", xy=(550*np.cos(a2), 550*np.sin(a2)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_SHADOW2, lw=2.4), zorder=8)
    ax.text(0, L, f"θ={theta:.0f}°  δ={delta:.0f}°  (double-oblique)",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    # ── Axes ─────────────────────────────────────────────────────
    view = max(wx, wy) * 1.6 + 300
    ax.set_xlim(-view, view); ax.set_ylim(-view, view + 400)
    ax.set_xlabel("x [nm]", color="white", fontsize=10)
    ax.set_ylabel("y [nm]", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#555")
    ax.grid(True, color="#333", alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8, facecolor="#2D2D44",
              labelcolor="white", framealpha=0.9)

    r = res
    status = ("OPEN (line pinched off)" if r["area_nm2"] <= 0
              else f"OK: {wnx:.0f} × {wny:.0f} nm")
    info = (f"w_narrow = w_open − h·sinδ/tanθ\n"
            f"shrink = {r['shrink_nm']:.0f} nm  (h={r['h_nm']:.0f} nm)\n"
            f"w_narrow_x = {wnx:.0f} nm  ·  w_narrow_y = {wny:.0f} nm\n"
            f"Area = {r['area_nm2']:.0f} nm²  ·  Ic ≈ {r['ic_estimate_uA']:.3f} µA\n"
            f"{status}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top",
            fontsize=8.5, color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#2D2D44", ec=C_JJ, lw=1.2))
    ax.set_title("Manhattan double-oblique — top view",
                 color="white", fontsize=11, fontweight="bold")
    fig.patch.set_facecolor(C_BG)
    fig.tight_layout()
    return fig
