"""
cross_section.py  (v5)
======================
5-panel process cross-section with correct Dolan bridge geometry.

Dolan bridge cross-section is shown along the EVAPORATION direction (x-z plane),
cutting through the middle of the bridge (y=0).

Structure in this cross-section:
  ████ PMMA ████   open gap   ████ PMMA ████
  ████ PMMA ████─ ─bridge─ ─ ████ PMMA ████   ← bridge spans the gap in PMMA
  ─ ─ MMA ─ ─ ─  air gap  ─ ─ MMA ─ ─ ─ ─   ← MMA undercut → wider air gap
  ──────────────── wafer ──────────────────

Bridge hangs at z ∈ [t_mma, t_mma + t_pmma]:
  - bottom face at z = t_mma (suspended over MMA air gap)
  - top face at z = t_mma + t_pmma (= resist top surface)
  - bridge sides are open (side trenches): metal can coat bridge side faces

Evaporation deposits:
  - Each beam comes through the side trench opening (not from above).
    In the x-z cross-section, the beam arrives at angle θ from normal,
    and the "top" of the resist is at z = t_mma + t_pmma.
  - Shadow of bridge right edge (at x=+L/2) blocks the beam for Evap1 (θ<0).
  - Metal that hits the bridge underside / side face stops there (dark colour).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from process_engine import ProcessParams, shadow_vector
from junction_area import compute_junction_area

# ── Colours ──────────────────────────────────────────────────────
C_SUB        = "#546E7A"
C_MMA        = "#FFF9C4"
C_PMMA       = "#FFB300"
C_BRIDGE     = "#66BB6A"   # bridge slab (green)
C_AIRGAP     = "#E3F2FD"   # MMA air gap (light blue)

C_M1_FLOOR   = "#42A5F5"   # Al₁ floor / bridge top
C_M1_TOP     = "#1565C0"   # Al₁ on resist outer top
C_M1_WALL    = "#0D47A1"   # Al₁ on sidewall / bridge underside (stopped)

C_M2_FLOOR   = "#EF5350"   # Al₂ floor beside Al₁
C_M2_JJ      = "#E53935"   # Al₂ on AlOx (junction)
C_M2_TOP     = "#B71C1C"   # Al₂ on resist outer top
C_M2_WALL    = "#4A148C"   # Al₂ on sidewall / face (stopped)

C_ALOX_LINE  = "#CE93D8"
C_ALOX_FILL  = "#F3E5F5"


def draw_cross_section(params: ProcessParams) -> plt.Figure:
    p   = params
    res = compute_junction_area(p)
    is_dolan = (p.mode == "Dolan bridge")

    t1, t2   = p.t_metal1, p.t_metal2
    t_alox   = max(t1 * 0.07, 3)
    t_mma    = p.t_mma
    t_pmma   = p.t_pmma
    t_tot    = p.t_resist

    sx1, sy1 = res["sx1"], res["sy1"]
    sx2, sy2 = res["sx2"], res["sy2"]

    if is_dolan:
        # x-direction openings
        L        = p.bridge_len
        pmma_hx  = L / 2                  # PMMA x half-gap (= bridge slab x half-span)
        mma_hx   = L / 2 + p.undercut     # MMA x half-gap

        # bridge z-range
        bridge_z0 = t_mma                  # bottom of bridge = top of MMA
        bridge_z1 = t_mma + t_pmma         # top of bridge = top of resist

        ev1_x_lo = res["ev1_x_lo"];  ev1_x_hi = res["ev1_x_hi"]
        ev2_x_lo = res["ev2_x_lo"];  ev2_x_hi = res["ev2_x_hi"]
        pmma_half = pmma_hx
        mma_half  = mma_hx
    else:
        pmma_half  = p.manhattan_wx / 2
        mma_half   = p.manhattan_wx / 2 + p.undercut
        bridge_z0 = bridge_z1 = 0;  L = 0
        ev1_x_lo = res["ev1_x_lo"];  ev1_x_hi = res["ev1_x_hi"]
        ev2_x_lo = res["ev2_x_lo"];  ev2_x_hi = res["ev2_x_hi"]

    r_outer  = mma_half + 700
    xlim     = (-r_outer - 120, r_outer + 320)
    ylim     = (-100, t_tot + 640)
    ov_lo    = max(ev1_x_lo, ev2_x_lo)
    ov_hi    = min(ev1_x_hi, ev2_x_hi)
    has_jj   = (ov_lo < ov_hi)

    jxc  = (ov_lo + ov_hi) / 2 if has_jj else 0
    zo_w = max((ov_hi - ov_lo) * 6 + 300, 600) if has_jj else 600

    # ── 5-panel figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor("#E4E4E4")
    gs = fig.add_gridspec(2, 6, hspace=0.40, wspace=0.30,
                          left=0.03, right=0.98, top=0.93, bottom=0.05)
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    ax_D = fig.add_subplot(gs[1, 0:4])
    ax_E = fig.add_subplot(gs[1, 4:6])

    def _setup(ax, title, dark=False):
        bg = "#1A1A2E" if dark else "#FAFAFA"
        tc = "#CCCCCC" if dark else "black"
        ax.set_facecolor(bg)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=tc, pad=5)
        ax.set_xlabel("x [nm]", fontsize=8, color=tc)
        ax.set_ylabel("z [nm]", fontsize=8, color=tc)
        ax.tick_params(labelsize=7.5, colors=tc)
        ax.grid(True, alpha=0.15, lw=0.6, color="#888" if dark else "#ccc")
        if dark:
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")

    _setup(ax_A, "A  After Evaporation 1")
    _setup(ax_B, "B  Oxidation  (Al₁ exposed surfaces → AlOx)")
    _setup(ax_C, "C  After Evaporation 2")
    _setup(ax_D, "D  After Lift-off", dark=True)
    _setup(ax_E, "E  Junction close-up  (Al₁ / AlOx / Al₂)")

    for ax in [ax_A, ax_B, ax_C]:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax_D.set_xlim(*xlim)
    ax_D.set_ylim(-100, t1 + t2 + t_alox + 220)
    ax_E.set_xlim(jxc - zo_w/2, jxc + zo_w/2)
    ax_E.set_ylim(-30, t1 + t2 + t_alox + 120)
    ax_E.set_xlabel("x [nm]  (junction region)", fontsize=8)

    # ── Draw each panel ───────────────────────────────────────────
    kw = dict(p=p, pmma_half=pmma_half, mma_half=mma_half, r_outer=r_outer,
              t_mma=t_mma, t_pmma=t_pmma, t_tot=t_tot, t1=t1, t2=t2,
              t_alox=t_alox, is_dolan=is_dolan,
              bridge_z0=bridge_z0, bridge_z1=bridge_z1,
              ev1_x_lo=ev1_x_lo, ev1_x_hi=ev1_x_hi,
              ev2_x_lo=ev2_x_lo, ev2_x_hi=ev2_x_hi,
              ov_lo=ov_lo, ov_hi=ov_hi, sx1=sx1, sx2=sx2,
              xlim=xlim)

    for ax in [ax_A, ax_B, ax_C]:
        _substrate(ax, xlim)
        _resist(ax, **kw)

    # Panel A
    _evap1(ax_A, show_alox=False, **kw)
    _arrow(ax_A, p.angle1, p.phi1, t_tot, xlim, C_M1_FLOOR,
           f"Evap 1  θ={p.angle1}° φ={p.phi1}°")

    # Panel B (oxidation)
    _evap1(ax_B, show_alox=True, **kw)
    ax_B.text(0, t_tot + 490,
              "O₂ → AlOx on all exposed Al₁",
              ha="center", fontsize=8, color="#6A1B9A",
              bbox=dict(boxstyle="round,pad=0.4", fc="#F3E5F5",
                        ec=C_ALOX_LINE, lw=1.4))

    # Panel C
    _evap1(ax_C, show_alox=True, **kw)
    _evap2(ax_C, **kw)
    _arrow(ax_C, p.angle1, p.phi1, t_tot, xlim, C_M1_FLOOR,
           f"Evap 1  θ={p.angle1}° φ={p.phi1}°", offset=0)
    _arrow(ax_C, p.angle2, p.phi2, t_tot, xlim, C_M2_FLOOR,
           f"Evap 2  θ={p.angle2}° φ={p.phi2}°", offset=1)

    # Panel D
    _substrate(ax_D, xlim)
    _liftoff(ax_D, **kw)

    # Panel E
    _closeup(ax_E, res=res, **kw)

    # ── Title ─────────────────────────────────────────────────────
    if is_dolan:
        geom_str = (f"bridge_len={p.bridge_len} nm  bridge_w={p.bridge_w} nm")
    else:
        geom_str = (f"x-arm {p.manhattan_wx} nm  y-arm {p.manhattan_wy} nm")

    fig.suptitle(
        f"Process cross-section — {p.mode}  ·  "
        f"PMMA {t_pmma} nm / MMA {t_mma} nm  undercut {p.undercut} nm  ·  "
        f"{geom_str}  ·  "
        f"θ₁={p.angle1}° φ₁={p.phi1}° / θ₂={p.angle2}° φ₂={p.phi2}°",
        fontsize=9.5, fontweight="bold"
    )
    return fig


# ════════════════════════════════════════════════════════════════
# Drawing helpers
# ════════════════════════════════════════════════════════════════

def _substrate(ax, xlim):
    ax.add_patch(patches.Rectangle(
        (xlim[0], -80), xlim[1]-xlim[0], 80,
        color=C_SUB, zorder=1))
    ax.text((xlim[0]+xlim[1])/2, -40, "Si / SiO₂ substrate",
            ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")


def _resist(ax, p, pmma_half, mma_half, r_outer,
            t_mma, t_pmma, t_tot, is_dolan,
            bridge_z0, bridge_z1, **kw):
    """Draw bilayer resist + bridge structure."""
    for sign in [-1, 1]:
        xo = sign * (r_outer + 100)
        # MMA block (with undercut zone)
        xi_mma = sign * mma_half
        ax.add_patch(patches.Rectangle(
            (min(xi_mma, xo), 0), abs(xo - xi_mma), t_mma,
            color=C_MMA, alpha=0.88, zorder=2,
            label="MMA" if sign == -1 else None))
        # Undercut zone
        xi_pmma = sign * pmma_half
        ax.add_patch(patches.Rectangle(
            (min(xi_mma, xi_pmma), 0), abs(xi_mma - xi_pmma), t_mma,
            fc="none", ec="#F57F17", ls="--", lw=1.2, zorder=5,
            label="Undercut" if sign == -1 else None))
        # PMMA block
        ax.add_patch(patches.Rectangle(
            (min(xi_pmma, xo), t_mma), abs(xo - xi_pmma), t_pmma,
            color=C_PMMA, alpha=0.85, zorder=2,
            label="PMMA" if sign == -1 else None))

    ax.text(-(r_outer+65), t_mma/2,          "MMA",  ha="right", fontsize=7, color="#5D4037")
    ax.text(-(r_outer+65), t_mma + t_pmma/2, "PMMA", ha="right", fontsize=7, color="#E65100")

    if is_dolan:
        # MMA air gap (light blue fill in the MMA opening)
        ax.add_patch(patches.Rectangle(
            (-mma_half, 0), 2*mma_half, t_mma,
            fc=C_AIRGAP, ec="#90CAF9", ls=":", lw=1.0, zorder=3,
            label="MMA air gap"))
        # Bridge slab: spans PMMA opening in x, sits at z ∈ [t_mma, t_mma+t_pmma]
        ax.add_patch(patches.Rectangle(
            (-pmma_half, bridge_z0), 2*pmma_half, bridge_z1 - bridge_z0,
            color=C_BRIDGE, alpha=0.88, zorder=6,
            label="Bridge (PMMA slab)"))
        ax.text(0, bridge_z0 + (bridge_z1-bridge_z0)/2, "bridge",
                ha="center", va="center", fontsize=7,
                color="#1B5E20", fontweight="bold")
        # Dimension: bridge length
        ax.annotate("", xy=(-pmma_half, -58), xytext=(pmma_half, -58),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.1))
        ax.text(0, -65, f"bridge_len={int(2*pmma_half)}nm",
                ha="center", va="top", fontsize=7, color="#333")

    # Dimension bar (right)
    xr = r_outer + 145
    ax.annotate("", xy=(xr,0), xytext=(xr,t_mma),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1))
    ax.text(xr+20, t_mma/2, f"MMA\n{t_mma}nm", va="center", fontsize=7, color="#795548")
    ax.annotate("", xy=(xr,t_mma), xytext=(xr,t_tot),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1))
    ax.text(xr+20, t_mma+t_pmma/2, f"PMMA\n{t_pmma}nm", va="center", fontsize=7, color="#E65100")


def _wall_coat(ax, x, z_lo, z_hi, thickness, color, alpha, zorder=7):
    """Thin metal coating on a vertical wall surface (stops at wall)."""
    sign = 1 if x >= 0 else -1
    ax.add_patch(patches.Rectangle(
        (x - sign*thickness, z_lo),
        sign * (-thickness) if sign > 0 else thickness,
        z_hi - z_lo,
        color=color, alpha=alpha, zorder=zorder))


def _evap1(ax, p, pmma_half, mma_half, r_outer,
           t_mma, t_pmma, t_tot, t1, t_alox, is_dolan,
           bridge_z0, bridge_z1,
           ev1_x_lo, ev1_x_hi, sx1,
           show_alox, **kw):
    sw = max(t1 * 0.35, 5)   # sidewall thickness for display

    # Resist top (outer electrodes)
    for sign in [-1, 1]:
        x0 = sign * mma_half; x1 = sign * (r_outer + 100)
        ax.add_patch(patches.Rectangle(
            (min(x0,x1), t_tot), abs(x1-x0), t1,
            color=C_M1_TOP, alpha=0.85, zorder=5,
            label="Al₁ (resist top)" if sign == -1 else None))

    # Floor deposit (inside the MMA gap, past the bridge shadow)
    if ev1_x_lo < ev1_x_hi:
        ax.add_patch(patches.Rectangle(
            (ev1_x_lo, 0), ev1_x_hi - ev1_x_lo, t1,
            color=C_M1_FLOOR, alpha=0.90, zorder=6, label="Al₁ (floor)"))

    if is_dolan:
        # Bridge underside coating (beam hits bottom face of bridge → stops)
        # The beam from Evap1 (θ₁<0 → from right) hits bridge underside
        # at x positions covered by bridge shadow on floor
        # Shown as thin coat on bridge bottom face
        bx0 = ev1_x_hi  # right edge of floor deposit
        bx1 = pmma_half  # right edge of bridge
        if bx0 < bx1:
            ax.add_patch(patches.Rectangle(
                (bx0, bridge_z0), bx1 - bx0, sw,
                color=C_M1_WALL, alpha=0.65, zorder=8,
                label="Al₁ (bridge underside)"))
        # Bridge side face (right side, if beam comes from left)
        # Coat the vertical face at x = ±pmma_half
        for bsign in [-1, 1]:
            bfx = bsign * pmma_half
            if (sx1 > 0 and bsign > 0) or (sx1 < 0 and bsign < 0) or sx1 == 0:
                _wall_coat(ax, bfx, bridge_z0, bridge_z1, sw,
                           C_M1_WALL, alpha=0.55, zorder=8)
        # Bridge top
        ax.add_patch(patches.Rectangle(
            (-pmma_half, bridge_z1), 2*pmma_half, t1,
            color=C_M1_FLOOR, alpha=0.88, zorder=8,
            label="Al₁ (bridge top)"))

    # PMMA inner sidewall (z: t_mma to t_tot)
    for sign in [-1, 1]:
        wx = sign * pmma_half
        if (sx1 > 0 and sign > 0) or (sx1 < 0 and sign < 0) or sx1 == 0:
            _wall_coat(ax, wx, t_mma, t_tot, sw,
                       C_M1_WALL, alpha=0.70, zorder=7)
    # MMA inner sidewall (z: 0 to t_mma) — only outer MMA wall
    for sign in [-1, 1]:
        wx = sign * mma_half
        if (sx1 > 0 and sign > 0) or (sx1 < 0 and sign < 0):
            _wall_coat(ax, wx, 0, t_mma, sw,
                       C_M1_WALL, alpha=0.50, zorder=7)

    # AlOx on exposed Al₁ (Panel B+)
    if show_alox:
        if ev1_x_lo < ev1_x_hi:
            ax.add_patch(patches.Rectangle(
                (ev1_x_lo, t1), ev1_x_hi-ev1_x_lo, t_alox,
                color=C_ALOX_LINE, alpha=0.60, zorder=9, label="AlOx"))
        for sign in [-1, 1]:
            x0 = sign * mma_half; x1 = sign * (r_outer+100)
            ax.add_patch(patches.Rectangle(
                (min(x0,x1), t_tot+t1), abs(x1-x0), t_alox,
                color=C_ALOX_LINE, alpha=0.40, zorder=8))
        if is_dolan:
            ax.add_patch(patches.Rectangle(
                (-pmma_half, bridge_z1+t1), 2*pmma_half, t_alox,
                color=C_ALOX_LINE, alpha=0.50, zorder=9))

    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85,
              handlelength=1.1, labelspacing=0.3)


def _evap2(ax, p, pmma_half, mma_half, r_outer,
           t_mma, t_pmma, t_tot, t1, t2, t_alox, is_dolan,
           bridge_z0, bridge_z1,
           ev1_x_lo, ev1_x_hi, ev2_x_lo, ev2_x_hi,
           ov_lo, ov_hi, sx2, **kw):
    sw = max(t2 * 0.35, 5)

    # Resist top: Evap2 on top of Al₁ + AlOx
    for sign in [-1, 1]:
        x0 = sign * mma_half; x1 = sign * (r_outer+100)
        ax.add_patch(patches.Rectangle(
            (min(x0,x1), t_tot+t1+t_alox), abs(x1-x0), t2,
            color=C_M2_TOP, alpha=0.82, zorder=6))

    # Junction region: Evap2 on AlOx
    if ov_lo < ov_hi:
        ax.plot([ov_lo, ov_hi], [t1+t_alox, t1+t_alox],
                color=C_ALOX_LINE, lw=2.5, zorder=11, solid_capstyle="butt",
                label="AlOx / Al₂ interface")
        ax.add_patch(patches.Rectangle(
            (ov_lo, t1+t_alox), ov_hi-ov_lo, t2,
            color=C_M2_JJ, alpha=0.90, zorder=10,
            label="Al₂ (junction, on AlOx)"))

    # Floor beside Al₁
    for lo, hi in [(ev2_x_lo, min(ev2_x_hi, ov_lo)),
                   (max(ev2_x_lo, ov_hi), ev2_x_hi)]:
        if lo < hi:
            ax.add_patch(patches.Rectangle(
                (lo, 0), hi-lo, t2,
                color=C_M2_FLOOR, alpha=0.80, zorder=7,
                label="Al₂ (floor, beside Al₁)"))

    # Exposed face of Al₁: Evap2 coats it (dark red)
    for face_x, face_dir in [(ev1_x_hi, +1), (ev1_x_lo, -1)]:
        if (sx2 > 0 and face_dir > 0) or (sx2 < 0 and face_dir < 0):
            x0 = face_x if face_dir > 0 else face_x - sw
            ax.add_patch(patches.Rectangle(
                (x0, t_alox), sw, t1,
                color=C_M2_WALL, alpha=0.60, zorder=9,
                label="Al₂ (Al₁ face)"))

    if is_dolan:
        # Bridge underside: Evap2 coats the region shadowed by bridge for Evap2
        bx0 = ev2_x_lo; bx1 = -pmma_half
        if bx1 < bx0:   # Evap2 comes from left → right
            bx0, bx1 = pmma_half, ev2_x_hi
        if bx0 < bx1:
            ax.add_patch(patches.Rectangle(
                (bx0, bridge_z0+t_alox), bx1-bx0, sw,
                color=C_M2_WALL, alpha=0.60, zorder=9))
        # Bridge side face
        for bsign in [-1, 1]:
            bfx = bsign * pmma_half
            if (sx2 > 0 and bsign > 0) or (sx2 < 0 and bsign < 0) or sx2 == 0:
                _wall_coat(ax, bfx, bridge_z0, bridge_z1, sw,
                           C_M2_WALL, alpha=0.50, zorder=9)
        # Bridge top: Evap2 on AlOx on Al₁
        ax.plot([-pmma_half, pmma_half], [bridge_z1+t1+t_alox]*2,
                color=C_ALOX_LINE, lw=2, zorder=11, solid_capstyle="butt")
        ax.add_patch(patches.Rectangle(
            (-pmma_half, bridge_z1+t1+t_alox), 2*pmma_half, t2,
            color=C_M2_JJ, alpha=0.88, zorder=10))

    # PMMA sidewall (Evap2 stopped by wall)
    for sign in [-1, 1]:
        wx = sign * pmma_half
        if (sx2 > 0 and sign > 0) or (sx2 < 0 and sign < 0) or sx2 == 0:
            _wall_coat(ax, wx, t_mma, t_tot, sw,
                       C_M2_WALL, alpha=0.65, zorder=8)
    for sign in [-1, 1]:
        wx = sign * mma_half
        if (sx2 > 0 and sign > 0) or (sx2 < 0 and sign < 0):
            _wall_coat(ax, wx, 0, t_mma, sw,
                       C_M2_WALL, alpha=0.45, zorder=8)

    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85,
              handlelength=1.0, labelspacing=0.25)


def _liftoff(ax, p, mma_half, r_outer, t1, t2, t_alox,
             ev1_x_lo, ev1_x_hi, ev2_x_lo, ev2_x_hi,
             ov_lo, ov_hi, xlim, **kw):
    # Al₁ floor
    if ev1_x_lo < ev1_x_hi:
        ax.add_patch(patches.Rectangle(
            (ev1_x_lo, 0), ev1_x_hi-ev1_x_lo, t1,
            color=C_M1_FLOOR, alpha=0.92, zorder=4, label="Al₁"))
    # Junction
    if ov_lo < ov_hi:
        ax.add_patch(patches.Rectangle(
            (ov_lo, t1), ov_hi-ov_lo, t_alox,
            color=C_ALOX_LINE, alpha=0.75, zorder=6, label="AlOx"))
        ax.add_patch(patches.Rectangle(
            (ov_lo, t1+t_alox), ov_hi-ov_lo, t2,
            color=C_M2_JJ, alpha=0.92, zorder=7, label="Al₂ (JJ)"))
    # Al₂ beside
    for lo, hi in [(ev2_x_lo, min(ev2_x_hi, ov_lo)),
                   (max(ev2_x_lo, ov_hi), ev2_x_hi)]:
        if lo < hi:
            ax.add_patch(patches.Rectangle(
                (lo, 0), hi-lo, t2,
                color=C_M2_FLOOR, alpha=0.80, zorder=5))
    # Electrodes
    for sign in [-1, 1]:
        x0 = sign * mma_half; x1 = sign * (r_outer+100)
        ax.add_patch(patches.Rectangle(
            (min(x0,x1),0), abs(x1-x0), t1,
            color=C_M1_FLOOR, alpha=0.88, zorder=4))
        ax.add_patch(patches.Rectangle(
            (min(x0,x1),t1+t_alox), abs(x1-x0), t2,
            color=C_M2_FLOOR, alpha=0.80, zorder=4))
    if ov_lo < ov_hi:
        ax.annotate("Josephson\njunction",
                    xy=((ov_lo+ov_hi)/2, t1+t_alox),
                    xytext=((ov_lo+ov_hi)/2, t1+t2+t_alox+80),
                    ha="center", fontsize=9, color=C_ALOX_LINE, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_ALOX_LINE, lw=1.5))
    ax.legend(loc="upper right", fontsize=8, facecolor="#2A2A3E",
              labelcolor="white", framealpha=0.9)
    ax.tick_params(colors="#CCCCCC")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")


def _closeup(ax, res, t1, t2, t_alox, ev1_x_lo, ev1_x_hi,
             ev2_x_lo, ev2_x_hi, ov_lo, ov_hi, **kw):
    t_alox_d = max(t_alox, 5)

    if ev1_x_lo < ev1_x_hi:
        ax.add_patch(patches.Rectangle(
            (ev1_x_lo, 0), ev1_x_hi-ev1_x_lo, t1,
            color=C_M1_FLOOR, alpha=0.90, zorder=4, label="Al₁"))

    for lo, hi in [(ev2_x_lo, min(ev2_x_hi, ov_lo)),
                   (max(ev2_x_lo, ov_hi), ev2_x_hi)]:
        if lo < hi:
            ax.add_patch(patches.Rectangle(
                (lo, 0), hi-lo, t2,
                color=C_M2_FLOOR, alpha=0.70, zorder=4))

    if ov_lo < ov_hi:
        ax.add_patch(patches.Rectangle(
            (ov_lo, t1), ov_hi-ov_lo, t_alox_d,
            color=C_ALOX_LINE, alpha=0.75, zorder=7, label="AlOx"))
        ax.plot([ov_lo,ov_hi],[t1,t1], color=C_ALOX_LINE, lw=3, zorder=8,
                solid_capstyle="butt")
        ax.plot([ov_lo,ov_hi],[t1+t_alox_d,t1+t_alox_d],
                color=C_ALOX_LINE, lw=3, zorder=8, solid_capstyle="butt")
        ax.add_patch(patches.Rectangle(
            (ov_lo, t1+t_alox_d), ov_hi-ov_lo, t2,
            color=C_M2_JJ, alpha=0.90, zorder=6, label="Al₂"))
        ax.add_patch(patches.Rectangle(
            (ov_lo, 0), ov_hi-ov_lo, t1+t_alox_d+t2,
            color=C_ALOX_LINE, alpha=0.06, zorder=3))
        ax.annotate("", xy=(ov_lo,-22), xytext=(ov_hi,-22),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.2))
        ax.text((ov_lo+ov_hi)/2, -26,
                f"Jxn\n{ov_hi-ov_lo:.0f} nm",
                ha="center", va="top", fontsize=8, color="#444")

    xr = ax.get_xlim()[1] - 15
    ax.annotate("", xy=(xr,0), xytext=(xr,t1),
                arrowprops=dict(arrowstyle="<->", color=C_M1_FLOOR, lw=1.1))
    ax.text(xr+6, t1/2, f"d₁={t1}nm", va="center", fontsize=7.5, color=C_M1_FLOOR)
    ax.annotate("", xy=(xr,t1), xytext=(xr,t1+t_alox_d),
                arrowprops=dict(arrowstyle="<->", color=C_ALOX_LINE, lw=1.1))
    ax.text(xr+6, t1+t_alox_d/2, "AlOx", va="center", fontsize=7, color=C_ALOX_LINE)
    ax.annotate("", xy=(xr,t1+t_alox_d), xytext=(xr,t1+t_alox_d+t2),
                arrowprops=dict(arrowstyle="<->", color=C_M2_JJ, lw=1.1))
    ax.text(xr+6, t1+t_alox_d+t2/2, f"d₂={t2}nm",
            va="center", fontsize=7.5, color=C_M2_JJ)

    info = (f"Area = {res['area_nm2']:.0f} nm²\n"
            f"Overlap x = {res['overlap_x_nm']:.1f} nm\n"
            f"Overlap y = {res['overlap_y_nm']:.1f} nm\n"
            f"Ic ≈ {res['ic_estimate_uA']:.3f} µA")
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top",
            fontsize=8.5, color="#1A237E",
            bbox=dict(boxstyle="round,pad=0.4", fc="#E8EAF6",
                      ec="#3949AB", lw=1.2))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _arrow(ax, theta, phi, t_tot, xlim, color, label, offset=0):
    arr = 270
    dx = -arr * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    dz = -arr * np.cos(np.radians(theta))
    xc = xlim[0]*0.36 + offset * 0.44*(xlim[1]-xlim[0])
    z0 = t_tot + 400
    ax.annotate("", xy=(xc+dx, z0+dz), xytext=(xc, z0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2))
    ax.text(xc, z0+16, label, ha="center", va="bottom", fontsize=7.5,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=color, lw=0.8, alpha=0.90))
