"""
phi_cross_section.py
====================
Cross-section and top-view figures for phi-dependent evaporation.

Figures produced:
  1. draw_phi_cross_section()  — side-view cross-section for each evaporation,
                                 showing the φ-shifted shadow in both x and y.
  2. draw_junction_topview()   — top-down view of the wafer surface showing
                                 the two deposited rectangles and their overlap
                                 (the junction), including tilt angle annotation.
  3. draw_phi_scan()           — area & tilt vs φ sweep plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
import copy

from process_engine import ProcessParams, shadow_vector
from junction_area import compute_junction_area

# ── Color palette ────────────────────────────────────────────────
C_SUBSTRATE = "#B0BEC5"
C_RESIST    = "#FFF176"
C_BRIDGE    = "#FFE0B2"
C_METAL1    = "#64B5F6"
C_METAL2    = "#EF9A9A"
C_OVERLAP   = "#CE93D8"
C_ARROW     = "#37474F"


# ════════════════════════════════════════════════════════════════
# 1.  Side-view cross-section (x–z plane, φ=0 reference plane)
# ════════════════════════════════════════════════════════════════
def draw_phi_cross_section(params: ProcessParams) -> plt.Figure:
    """
    Two-panel cross-section (evap 1 | evap 2).
    For φ ≠ 0 the shadow vector has both x and y components;
    this view shows the x–z slice (perpendicular to bridge).
    A note shows the y-shift due to φ.
    """
    p = params
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    resist_half  = p.bridge_width / 2 + 400
    opening_half = p.bridge_width / 2
    uc_half      = opening_half + p.undercut
    bridge_y0    = p.bridge_gap
    bridge_y1    = p.bridge_gap + 60
    resist_y1    = p.t_resist

    evaps = [
        (axes[0], p.angle1, p.phi1, p.t_metal1, C_METAL1, "Evaporation 1"),
        (axes[1], p.angle2, p.phi2, p.t_metal2, C_METAL2, "Evaporation 2"),
    ]

    for ax, theta, phi, t_metal, c_metal, title in evaps:
        sx, sy = shadow_vector(p.t_resist, theta, phi)
        ax.set_facecolor("#F5F5F5")

        # Substrate
        ax.add_patch(patches.Rectangle(
            (-resist_half-100, -80), (resist_half+100)*2, 80,
            color=C_SUBSTRATE, zorder=1))
        ax.text(0, -40, "Si / SiO₂ substrate", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

        # Resist blocks + undercut
        for sign in [-1, 1]:
            x_inner = sign * uc_half
            x_outer = sign * (resist_half + 100)
            ax.add_patch(patches.Rectangle(
                (min(x_inner, x_outer), 0), abs(x_outer - x_inner), resist_y1,
                color=C_RESIST, alpha=0.9, zorder=2,
                label="Resist" if sign == -1 else None))
            ax.add_patch(patches.Rectangle(
                (sign * opening_half if sign > 0 else sign * uc_half, 0),
                p.undercut * sign * (-1 if sign > 0 else 1), resist_y1,
                color="#FFCC02", alpha=0.4, zorder=3,
                label="Undercut" if sign == -1 else None))

        # Bridge
        ax.add_patch(patches.Rectangle(
            (-p.bridge_width/2, bridge_y0), p.bridge_width, bridge_y1-bridge_y0,
            color=C_BRIDGE, alpha=0.95, zorder=4, label="Bridge"))
        ax.text(0, (bridge_y0+bridge_y1)/2, "bridge",
                ha="center", va="center", fontsize=7, color="#6D4C41")

        # Metal on resist top
        for sign in [-1, 1]:
            x0, x1 = sign * uc_half, sign * (resist_half+100)
            ax.add_patch(patches.Rectangle(
                (min(x0,x1), resist_y1), abs(x1-x0), t_metal,
                color=c_metal, alpha=0.8, zorder=5))

        # Metal on wafer surface (x-component of shadow)
        jj_x0 = -uc_half - sx
        jj_x1 =  uc_half - sx
        if jj_x0 < jj_x1:
            ax.add_patch(patches.Rectangle(
                (jj_x0, 0), jj_x1-jj_x0, t_metal,
                color=c_metal, alpha=0.7, zorder=5, label="Floor deposit"))

        # Metal on bridge top
        ax.add_patch(patches.Rectangle(
            (-p.bridge_width/2, bridge_y1), p.bridge_width, t_metal,
            color=c_metal, alpha=0.8, zorder=6))

        # Evaporation arrow (in x-z plane)
        arr_x  = resist_half * 0.65 * np.sign(np.sin(np.radians(phi)) if phi != 0
                                               else np.tan(np.radians(theta)))
        arr_len = 260
        dx = -arr_len * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
        dz = -arr_len * np.cos(np.radians(theta))
        ax.annotate("", xy=(arr_x+dx, resist_y1+300+dz),
                    xytext=(arr_x, resist_y1+300),
                    arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=2))
        ax.text(arr_x, resist_y1+345,
                f"θ={theta}°, φ={phi}°",
                ha="center", va="bottom", fontsize=8.5, color=C_ARROW, fontweight="bold")

        # Dimension: resist thickness
        ax.annotate("", xy=(resist_half+150, resist_y1),
                    xytext=(resist_half+150, 0),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
        ax.text(resist_half+205, resist_y1/2,
                f"t={p.t_resist} nm", va="center", fontsize=8, color="#333")

        # Shadow x-offset annotation
        if abs(sx) > 5:
            ax.annotate("", xy=(jj_x0, -45), xytext=(-uc_half, -45),
                        arrowprops=dict(arrowstyle="<->", color="#e55", lw=1.2))
            ax.text((-uc_half + jj_x0)/2, -58,
                    f"sx={sx:.0f} nm", ha="center", fontsize=7.5, color="#c33")

        # y-shift note
        if abs(sy) > 1:
            ax.text(0, resist_y1+220,
                    f"y-shift along bridge: sy = {sy:+.1f} nm",
                    ha="center", fontsize=8, color="#5533aa",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#EDE7F6", ec="#7E57C2", lw=1))

        ax.set_xlim(-resist_half-200, resist_half+360)
        ax.set_ylim(-100, resist_y1+400)
        ax.set_xlabel("x [nm]", fontsize=10)
        ax.set_ylabel("z [nm]", fontsize=10)
        ax.set_title(f"{title}  (x–z cross-section)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")

    fig.suptitle("Cross-section with φ (azimuthal) evaporation angle",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# 2.  Junction top-view (x–y plane on wafer surface)
# ════════════════════════════════════════════════════════════════
def draw_junction_topview(params: ProcessParams) -> plt.Figure:
    """
    Top-down view of the wafer surface under the bridge opening.
    Shows deposit 1, deposit 2, and their intersection (junction).
    Annotates tilt angle and area.
    """
    p   = params
    res = compute_junction_area(p)

    sx1, sy1 = res["sx1"], res["sy1"]
    sx2, sy2 = res["sx2"], res["sy2"]
    uc_half   = res["uc_half_nm"]
    w         = p.bridge_width

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("#1C1C2E")
    ax.set_aspect("equal")

    # Deposit rectangles on wafer surface
    # Deposit 1: x ∈ [-uc_half-sx1, uc_half-sx1], y ∈ [-w/2+sy1, w/2+sy1]
    dep1 = _rect((-uc_half - sx1, -w/2 + sy1), (2*uc_half, w))
    dep2 = _rect((-uc_half - sx2, -w/2 + sy2), (2*uc_half, w))

    ax.add_patch(Polygon(dep1, closed=True, facecolor=C_METAL1,
                         edgecolor="#1565C0", lw=1.5, alpha=0.35, zorder=2,
                         label=f"Deposit 1  (θ={p.angle1}°, φ={p.phi1}°)"))
    ax.add_patch(Polygon(dep2, closed=True, facecolor=C_METAL2,
                         edgecolor="#B71C1C", lw=1.5, alpha=0.35, zorder=2,
                         label=f"Deposit 2  (θ={p.angle2}°, φ={p.phi2}°)"))

    # Junction overlap rectangle
    jrect = res["junction_rect"]
    if jrect:
        jx = [c[0] for c in jrect] + [jrect[0][0]]
        jy = [c[1] for c in jrect] + [jrect[0][1]]
        ax.fill(jx, jy, color=C_OVERLAP, alpha=0.85, zorder=4,
                label=f"Junction  A={res['area_nm2']:.0f} nm²")
        ax.plot(jx, jy, color="white", lw=1.5, zorder=5)

        # Tilt annotation arrow
        cx = (jrect[0][0] + jrect[1][0]) / 2
        cy = (jrect[0][1] + jrect[2][1]) / 2
        tilt = res["junction_tilt_deg"]
        arrow_len = w * 0.55
        dx = arrow_len * np.cos(np.radians(tilt))
        dy = arrow_len * np.sin(np.radians(tilt))
        ax.annotate("", xy=(cx+dx, cy+dy), xytext=(cx-dx, cy-dy),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.8))
        ax.text(cx + dx*1.15, cy + dy*1.15,
                f"α={tilt:.1f}°", color="white", fontsize=9, fontweight="bold",
                ha="center", va="center")

    # Bridge outline
    bridge_rect = _rect((-w/2, -w/2), (w, w))
    ax.add_patch(Polygon(bridge_rect, closed=True, facecolor="none",
                         edgecolor="#FFE082", lw=2, linestyle="--", zorder=6,
                         label="Bridge footprint"))

    # Undercut opening outline
    uc_rect = _rect((-uc_half, -w/2 - p.undercut), (2*uc_half, w + 2*p.undercut))
    ax.add_patch(Polygon(uc_rect, closed=True, facecolor="none",
                         edgecolor="#FFF176", lw=1.2, linestyle=":", zorder=6,
                         label="Undercut opening"))

    # Axes
    margin = uc_half * 0.6
    ax.set_xlim(-uc_half - margin, uc_half + margin)
    ax.set_ylim(-w/2 - p.undercut - margin, w/2 + p.undercut + margin)
    ax.set_xlabel("x [nm]  (perpendicular to bridge)", color="white", fontsize=10)
    ax.set_ylabel("y [nm]  (along bridge axis)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#555")
    ax.grid(True, color="#333", alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8, facecolor="#2D2D44",
              labelcolor="white", framealpha=0.9)

    # Summary text box
    info = (f"Area = {res['area_nm2']:.0f} nm²\n"
            f"Tilt α = {res['junction_tilt_deg']:.1f}°\n"
            f"Ic ≈ {res['ic_estimate_uA']:.3f} µA\n"
            f"Δy = sy₂−sy₁ = {sy2-sy1:.1f} nm")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8.5,
            color="white", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="#1E1E3A", ec="#7E57C2", lw=1.2))

    ax.set_title("Junction top-view (wafer surface, under bridge)",
                 color="white", fontsize=11, fontweight="bold")
    fig.patch.set_facecolor("#1C1C2E")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# 3.  φ parameter scan: area and tilt vs φ
# ════════════════════════════════════════════════════════════════
def draw_phi_scan(params: ProcessParams, which: str = "phi2") -> plt.Figure:
    """
    Sweeps φ₁ or φ₂ over [-90, 90]° and plots junction area and tilt angle.

    Parameters
    ----------
    params : base ProcessParams
    which  : "phi1" or "phi2" — which azimuthal angle to sweep
    """
    phi_vals = np.linspace(-90, 90, 200)
    areas, tilts, overlaps_x, overlaps_y = [], [], [], []

    for phi in phi_vals:
        p2 = copy.copy(params)
        setattr(p2, which, float(phi))
        r = compute_junction_area(p2)
        areas.append(r["area_nm2"])
        tilts.append(r["junction_tilt_deg"])
        overlaps_x.append(r["overlap_x_nm"])
        overlaps_y.append(r["overlap_y_nm"])

    areas      = np.array(areas)
    tilts      = np.array(tilts)
    overlaps_x = np.array(overlaps_x)
    overlaps_y = np.array(overlaps_y)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    phi_label = "φ₁" if which == "phi1" else "φ₂"
    fig.suptitle(f"φ scan: {phi_label} swept, other parameters fixed",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Junction area ────────────────────────────────────
    ax = axes[0]
    ax.plot(phi_vals, areas, lw=2.2, color=C_OVERLAP)
    ax.fill_between(phi_vals, areas, 0, alpha=0.15, color=C_OVERLAP)
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_ylabel("Junction area [nm²]", fontsize=10)
    ax.set_title("Junction area vs azimuthal angle")
    ax.grid(True, alpha=0.3)
    # mark current phi
    cur_phi = getattr(params, which)
    cur_res = compute_junction_area(params)
    ax.axvline(cur_phi, color="#FFE082", lw=1.5, ls="--", alpha=0.8)
    ax.scatter([cur_phi], [cur_res["area_nm2"]], color="#FFE082", zorder=5, s=60)

    # ── Panel 2: Overlap x and y ──────────────────────────────────
    ax = axes[1]
    ax.plot(phi_vals, overlaps_x, lw=2, color=C_METAL1, label="Overlap x (⊥ bridge)")
    ax.plot(phi_vals, overlaps_y, lw=2, color=C_METAL2, label="Overlap y (∥ bridge)")
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.fill_between(phi_vals, overlaps_x, 0, where=overlaps_x > 0,
                    alpha=0.1, color=C_METAL1)
    ax.fill_between(phi_vals, overlaps_y, 0, where=overlaps_y > 0,
                    alpha=0.1, color=C_METAL2)
    ax.set_ylabel("Overlap [nm]", fontsize=10)
    ax.set_title("x and y overlaps vs φ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(cur_phi, color="#FFE082", lw=1.5, ls="--", alpha=0.8)

    # ── Panel 3: Junction tilt ────────────────────────────────────
    ax = axes[2]
    ax.plot(phi_vals, tilts, lw=2, color="#80CBC4")
    ax.axhline(0, color="#aaa", lw=0.8, ls=":")
    ax.set_ylabel("Junction tilt α [°]", fontsize=10)
    ax.set_xlabel(f"{phi_label} [°]", fontsize=10)
    ax.set_title("Junction boundary tilt angle vs φ")
    ax.grid(True, alpha=0.3)
    ax.axvline(cur_phi, color="#FFE082", lw=1.5, ls="--", alpha=0.8,
               label=f"Current {phi_label}={cur_phi}°")
    ax.scatter([cur_phi], [cur_res["junction_tilt_deg"]],
               color="#FFE082", zorder=5, s=60)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ── helpers ──────────────────────────────────────────────────────
def _rect(origin, size):
    """Returns 4 corners of axis-aligned rectangle as numpy array."""
    x0, y0 = origin
    w,  h  = size
    return np.array([[x0, y0], [x0+w, y0], [x0+w, y0+h], [x0, y0+h]])
