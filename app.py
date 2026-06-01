"""
ShadowCast v5 — Josephson Junction Process Simulator
=====================================================
Correct Dolan bridge geometry (arxiv:2101.01453):
  - bridge_len × bridge_w parameters
  - Side trenches open (beam enters from the side)
  - bridge = PMMA slab suspended over MMA air gap
5-panel cross-section: Evap1 / Oxidation / Evap2 / Lift-off / Close-up
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, copy, os, sys

sys.path.insert(0, os.path.dirname(__file__))

from gds_parser import load_gds_polygons, list_layers
from process_engine import ProcessParams, shadow_vector
from cross_section import draw_cross_section
from phi_cross_section import draw_junction_topview, draw_phi_scan
from manhattan_check import manhattan_break_check
from top_view import draw_top_view
from junction_area import compute_junction_area
from deposition3d import simulate, junction_footprint
import voxel_view as vv

st.set_page_config(page_title="ShadowCast", page_icon="⚛️", layout="wide")
st.title("⚛️ ShadowCast — Josephson Junction Process Simulator")
st.caption(
    "Bilayer resist (PMMA/MMA) · θ & φ evaporation · Sidewall shadowing · "
    "Oxidation step · **Dolan bridge** (bridge_len × bridge_w, side trenches open) · "
    "**Manhattan cross** (φ₁ ⊥ φ₂)"
)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 GDS File")
    uploaded = st.file_uploader("Upload GDS / GDSII", type=["gds", "gdsii"])

    st.header("🔧 Process Parameters")
    mode = st.radio("Junction type", ["Dolan bridge", "Manhattan"], horizontal=True)

    st.subheader("Bilayer resist")
    st.caption("Recipe arxiv:2101.01453 — PMMA A-4 ≈250 nm / MMA EL-13 ≈900 nm")
    t_pmma   = st.slider("PMMA [nm]  (top, no undercut)",   100, 800, 250, 25)
    bridge_gap = st.slider("Bridge gap [nm]  (suspended height = shadow-defining)",
                           100, 1500, 900, 25,
                           help="Air-gap height under the Dolan bridge. "
                                "Junction overlap ≈ 2·gap·tanθ − bridge width. "
                                "Independent of the MMA layer thickness.")
    t_mma    = bridge_gap   # MMA bottom layer fills the gap region
    undercut = st.slider("MMA undercut u [nm]  (one-sided)",  0, 500, 150, 10)
    st.caption(f"Total resist: {t_pmma+bridge_gap} nm  ·  bridge gap (shadow height) = {bridge_gap} nm")

    st.subheader("Evaporation 1")
    angle1   = st.slider("Polar θ₁ [°]",     -60, 60, -24, 1)
    phi1     = st.slider("Azimuthal φ₁ [°]", -90, 90,   0, 1)
    t_metal1 = st.slider("Metal d₁ [nm]",     10, 200,  30, 5)

    if mode == "Dolan bridge":
        st.subheader("Evaporation 2")
        angle2   = st.slider("Polar θ₂ [°]",     -60, 60,  24, 1,
                             help="Dolan = uniaxial tilt: φ₂=φ₁, θ₂=−θ₁")
        phi2     = st.slider("Azimuthal φ₂ [°]", -90, 90,   0, 1)
        t_metal2 = st.slider("Metal d₂ [nm]",  10, 200, 30, 5)

        st.subheader("Geometry")
        st.markdown("Bridge slab dimensions:")
        bridge_len = st.slider("Bridge width [nm]  (evap direction x, shadow-defining)",
                               50, 2000, 250, 10)
        bridge_w   = st.slider("Bridge length [nm]  (junction width, y)",
                               50, 1000, 250, 10)
        t_shadow   = bridge_gap * np.tan(np.radians(max(abs(angle1), abs(angle2))))
        st.caption(
            f"Shadow projection = gap · tan θ = {t_shadow:.0f} nm  \n"
            f"Junction when bridge width **<** {2*t_shadow:.0f} nm  \n"
            f"(bridge_len < 2 · gap · tan θ  →  deposits meet under bridge)"
        )
        manhattan_wx = manhattan_wy = bridge_w
        manhattan_theta = 60.0; manhattan_delta = 15.0; manhattan_h = 1800.0
    else:
        st.subheader("Double-oblique evaporation")
        st.caption("Recipe arxiv:2605.19590 — θ ≈ 60°, δ ≈ 15–25°, h ≈ 1.8 µm")
        manhattan_theta = st.slider("Deposition tilt θ [°]  (from normal, shared)",
                                    30, 80, 60, 1)
        # Evaporation 1 and 2 are independent beams; default orthogonal (φ₁=0, φ₂=90).
        # φ₁ comes from the shared Evaporation-1 section above (default 0).
        st.subheader("Evaporation 2")
        phi2     = st.slider("Azimuthal φ₂ [°]", -90, 180, 90, 1,
                             help="Default 90° → perpendicular to Evap 1")
        t_metal2 = st.slider("Metal d₂ [nm]", 10, 200, 30, 5, key="m_d2")
        # Manhattan beams are tilted by the shared θ; reuse θ₁=θ₂=θ.
        angle1 = manhattan_theta
        angle2 = manhattan_theta
        # δ for Eq A6 = azimuth offset of each beam from its electrode line
        # (Evap1 line along x → offset |φ₁|; Evap2 line along y → offset |φ₂−90|).
        manhattan_delta = max(abs(phi1), abs(phi2 - 90.0), 1.0)
        manhattan_h     = st.slider("Imaging resist h [nm]", 500, 3000, 1800, 50)

        st.subheader("Geometry")
        st.markdown("Designed resist line openings (Manhattan crossing):")
        manhattan_wx = st.slider("x-arm opening wx [nm]", 100, 2000, 600, 10)
        manhattan_wy = st.slider("y-arm opening wy [nm]", 100, 2000, 600, 10)
        bridge_len = 700; bridge_w = 300
        _tan = np.tan(np.radians(manhattan_theta))
        _shrink = manhattan_h * np.sin(np.radians(manhattan_delta)) / _tan if _tan > 1e-9 else 0.0
        st.caption(
            f"Eq A6: w_narrow = w_open − h·sin δ / tan θ  \n"
            f"shrink = {_shrink:.0f} nm → w_narrow = "
            f"{max(manhattan_wx-_shrink,0):.0f} × {max(manhattan_wy-_shrink,0):.0f} nm"
        )

    st.divider()
    st.subheader("Display")
    show_shadow   = st.checkbox("Show shadow deposits (top view)", True)
    show_undercut = st.checkbox("Show undercut regions (top view)", True)

params = ProcessParams(
    t_pmma=t_pmma, t_mma=t_mma, bridge_gap=bridge_gap, undercut=undercut,
    angle1=angle1, phi1=phi1, t_metal1=t_metal1,
    angle2=angle2, phi2=phi2, t_metal2=t_metal2,
    bridge_len=bridge_len, bridge_w=bridge_w,
    manhattan_wx=manhattan_wx, manhattan_wy=manhattan_wy,
    manhattan_theta=manhattan_theta, manhattan_delta=manhattan_delta,
    manhattan_h=manhattan_h,
    mode=mode,
)
res = compute_junction_area(params)

# ─── 3D physical deposition engine (source of truth) ──────────────
@st.cache_data(show_spinner=False)
def _run_engine(ekey, _params):
    r = simulate(_params)
    jm, area, ox, oy = junction_footprint(r)
    return r, jm, area, ox, oy

ekey = (params.mode, params.t_pmma, params.t_mma, params.bridge_gap, params.undercut,
        params.angle1, params.phi1, params.t_metal1,
        params.angle2, params.phi2, params.t_metal2,
        params.bridge_len, params.bridge_w,
        params.manhattan_wx, params.manhattan_wy,
        params.manhattan_theta, params.manhattan_delta, params.manhattan_h)
with st.spinner("Running 3D shadow-evaporation engine..."):
    eng, eng_jm, eng_area, eng_ox, eng_oy = _run_engine(ekey, params)
# Engine-based critical current (jc = 10 kA/cm², Ambegaokar-Baratoff),
# consistent with junction_area.py:  Ic[µA] = area_nm2 · 1e-4
eng_ic = eng_area * 1e-4

# ─── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Cross-section",
    "🗺️ Top View",
    "🔄 φ Junction View",
    "🔍 Break Check",
    "📊 Junction Area",
])

# ═══ TAB 1: Cross-section ════════════════════════════════════════
with tab1:
    st.subheader("Cross-section  (3D voxel engine — x–z and y–z planes)")
    st.markdown(
        "Metal positions are computed by the **physical shadow-evaporation engine**: "
        "the real 3-D resist geometry is built as boxes and the tilted beam is "
        "ray-traced into a voxel grid.  Slice any plane below."
    )

    cc1, cc2 = st.columns([1, 2])
    with cc1:
        plane = st.radio("Slice plane", ["x–z", "y–z"], horizontal=True)
    with cc2:
        if plane == "x–z":
            ymin, ymax = float(eng.ys[0]), float(eng.ys[-1])
            slice_pos = st.slider("Slice at y [nm]", ymin, ymax, 0.0, eng.vox)
        else:
            xmin, xmax = float(eng.xs[0]), float(eng.xs[-1])
            slice_pos = st.slider("Slice at x [nm]", xmin, xmax, 0.0, eng.vox)

    _pl = "x-z" if plane == "x–z" else "y-z"

    _gR = float(eng.meta.get("grid_R", eng.meta["R"]))
    _ztop = float(eng.zs[-1])
    with st.expander("🔍 表示範囲 (View range)", expanded=False):
        vr1, vr2 = st.columns(2)
        cs_half = vr1.slider("Horizontal half-width [nm]", 100.0, _gR,
                             min(_gR, 800.0), 50.0, key="cs_half")
        cs_zmax = vr2.slider("Z max [nm]", 100.0, _ztop,
                             min(_ztop, 600.0), 50.0, key="cs_zmax")

    with st.spinner("Slicing voxel grid..."):
        st.markdown("**Process stages** — resist → evap 1 → oxidation → evap 2 → lift-off")
        figs = vv.render_stages(eng, _pl, slice_pos, eng_jm,
                                view_half=cs_half, zmax=cs_zmax)
        st.pyplot(figs, use_container_width=True)
        plt.close(figs)
        st.markdown("**Combined slice** (all layers, junction highlighted)")
        figc = vv.render_cross_section(eng, _pl, slice_pos, eng_jm,
                                       view_half=cs_half, zmax=cs_zmax)
        st.pyplot(figc, use_container_width=True)
        plt.close(figc)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voxel size", f"{eng.vox:.1f} nm")
    c2.metric("Junction area (engine)", f"{eng_area:.0f} nm²")
    c3.metric("Overlap x (engine)", f"{eng_ox:.0f} nm")
    c4.metric("Overlap y (engine)", f"{eng_oy:.0f} nm")

    if eng_area <= 0:
        if mode == "Manhattan":
            _t = np.tan(np.radians(params.manhattan_theta))
            _trans = params.manhattan_h * _t * np.sin(np.radians(params.manhattan_delta))
            st.error(
                f"❌ Open circuit (ray physics).  Transverse sidewall shadow "
                f"h·tanθ·sinδ ≈ {_trans:.0f} nm exceeds the linewidth.  \n"
                f"A junction forms physically only when h·tanθ·sinδ < linewidth "
                f"(roughly θ ≲ 45° or h ≲ 800 nm at w = 600 nm)."
            )
        else:
            st.error("❌ Open circuit — no metal overlap under the bridge.")
    else:
        st.success(f"✅ Junction formed: {eng_ox:.0f} × {eng_oy:.0f} nm (engine)")

    st.info(
        "**Colour guide:**  grey = substrate · tan = resist (PMMA/MMA) · "
        "blue = Al #1 · red = Al #2 · purple = AlOx · teal = junction (Al1∩Al2)"
    )

# ═══ TAB 2: Top View ═════════════════════════════════════════════
with tab2:
    st.subheader("Top View — engine floor map (process stages)")
    st.markdown(
        "Top-down view from the 3D voxel engine.  The resist opening shows the "
        "**undercut shelf** (pale) under the imaging resist (dark).  Stages: "
        "resist → evap 1 → oxidation → evap 2 → lift-off (junction highlighted)."
    )
    _gR2 = float(eng.meta.get("grid_R", eng.meta["R"]))
    with st.expander("🔍 表示範囲 (View range)", expanded=False):
        top_half = st.slider("Half-width [nm]", 100.0, _gR2,
                             min(_gR2, 1000.0), 50.0, key="top_half")

    with st.spinner("Rendering staged top view..."):
        figts = vv.render_top_stages(eng, eng_jm, view_half=top_half)
        st.pyplot(figts, use_container_width=True)
        plt.close(figts)

    st.markdown("**Final floor deposit** (combined)")
    with st.spinner("Rendering floor map..."):
        figt = vv.render_top_view(eng, eng_jm, view_half=top_half)
        st.pyplot(figt, use_container_width=True)
        plt.close(figt)

# ═══ TAB 3: φ Junction View ══════════════════════════════════════
with tab3:
    st.subheader("Junction Map (engine) + φ azimuth view")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Engine junction map** (floor deposit, JJ highlighted)")
        with st.spinner("Rendering..."):
            fig3 = vv.render_top_view(eng, eng_jm)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
    with col2:
        if mode == "Dolan bridge":
            st.markdown("**φ scan** (analytic estimate)")
            sweep = st.radio("Sweep", ["φ₂", "φ₁"], horizontal=True, key="phi_sw")
            wk = "phi2" if sweep == "φ₂" else "phi1"
            with st.spinner("Computing..."):
                fig4 = draw_phi_scan(params, which=wk)
                st.pyplot(fig4, use_container_width=True)
                plt.close(fig4)
        else:
            st.info(
                "For Manhattan, the two beams run at φ₁ / φ₂ (default 0° / 90°). "
                "Adjust them in the sidebar; the engine recomputes the overlap."
            )

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overlap x (engine)", f"{eng_ox:.0f} nm")
    c2.metric("Overlap y (engine)", f"{eng_oy:.0f} nm")
    c3.metric("Area (engine)",      f"{eng_area:.0f} nm²")
    c4.metric("Est. Ic (engine)",   f"{eng_ic:.3f} µA")

    if eng_area <= 0:
        st.error("❌ Open circuit — no junction overlap (engine).")
    elif min(eng_ox, eng_oy) < 2 * eng.vox:
        st.warning("⚠️ Tight overlap margin (near voxel resolution).")
    else:
        st.success("✅ Junction formed (engine).")

# ═══ TAB 4: Break Check ══════════════════════════════════════════
with tab4:
    st.subheader("Open Circuit / Short Circuit Check")

    ox, oy = eng_ox, eng_oy
    c1,c2,c3 = st.columns(3)
    c1.metric("Overlap x × y (engine)", f"{ox:.0f} × {oy:.0f} nm")
    c2.metric("Area (engine)", f"{eng_area:.0f} nm²")
    c3.metric("Est. Ic (engine)", f"{eng_ic:.3f} µA")

    if eng_area <= 0:
        if mode == "Dolan bridge":
            t_shadow_total = 2 * params.bridge_gap * np.tan(
                np.radians(max(abs(angle1), abs(angle2))))
            st.error(
                f"❌ Open circuit — no junction overlap (engine).  \n"
                f"Shadow projection: 2×gap×tan(θ) ≈ {t_shadow_total:.0f} nm  \n"
                f"bridge width must be **<** {t_shadow_total:.0f} nm for junction to form."
            )
        else:
            _t = np.tan(np.radians(params.manhattan_theta))
            _trans = params.manhattan_h * _t * np.sin(np.radians(params.manhattan_delta))
            st.error(
                f"❌ Open circuit (engine).  Transverse sidewall shadow "
                f"h·tanθ·sinδ ≈ {_trans:.0f} nm exceeds the linewidth — "
                f"reduce θ/h or increase the linewidth."
            )
    elif min(ox, oy) < 2 * eng.vox:
        st.warning("⚠️ Tight margin — near voxel resolution; check tolerances.")
    else:
        st.success(f"✅ Junction formed (engine): {ox:.0f} × {oy:.0f} nm")

    st.divider()
    st.subheader("Parameter Scan")
    st.caption("Fast analytic estimate for exploring trends. The headline "
               "open/short judgment above uses the 3D engine.")
    if mode == "Dolan bridge":
        scan_opts = ["θ₁", "θ₂", "φ₁", "φ₂",
                     "bridge_len", "bridge_w",
                     "MMA thickness", "Undercut u", "PMMA thickness"]
    else:
        scan_opts = ["θ tilt", "δ offset", "h resist",
                     "x-arm width", "y-arm width"]
    scan_p = st.selectbox("Scan parameter", scan_opts)
    scan_map = {
        "θ₁":           ("angle1",      np.linspace(-55,55,100),    "θ₁ [°]"),
        "θ₂":           ("angle2",      np.linspace(-55,55,100),    "θ₂ [°]"),
        "φ₁":           ("phi1",        np.linspace(-90,90,100),    "φ₁ [°]"),
        "φ₂":           ("phi2",        np.linspace(-90,90,100),    "φ₂ [°]"),
        "bridge_len":   ("bridge_len",  np.linspace(100,2000,100),  "bridge_len [nm]"),
        "bridge_w":     ("bridge_w",    np.linspace(50,800,100),    "bridge_w [nm]"),
        "Bridge gap":   ("bridge_gap",  np.linspace(100,1500,100),  "bridge gap [nm]"),
        "Undercut u":   ("undercut",    np.linspace(0,400,100),     "u [nm]"),
        "PMMA thickness":("t_pmma",     np.linspace(100,800,100),   "PMMA [nm]"),
        "θ tilt":       ("manhattan_theta", np.linspace(30,80,100),  "θ [°]"),
        "δ offset":     ("manhattan_delta", np.linspace(0,45,100),   "δ [°]"),
        "h resist":     ("manhattan_h",     np.linspace(500,3000,100),"h [nm]"),
        "x-arm width":  ("manhattan_wx",np.linspace(100,2000,100),   "wx [nm]"),
        "y-arm width":  ("manhattan_wy",np.linspace(100,2000,100),   "wy [nm]"),
    }
    attr, vals, xlabel = scan_map[scan_p]
    areas_s, tilts_s, ox_s, oy_s = [], [], [], []
    for v in vals:
        p2 = copy.copy(params); setattr(p2, attr, float(v))
        r2 = compute_junction_area(p2)
        areas_s.append(r2["area_nm2"])
        tilts_s.append(r2["junction_tilt_deg"])
        ox_s.append(r2["overlap_x_nm"])
        oy_s.append(r2["overlap_y_nm"])
    areas_s = np.array(areas_s)

    fig_s, axes_s = plt.subplots(3,1, figsize=(8,7), sharex=True)
    axes_s[0].plot(vals, areas_s, lw=2.2, color="#CE93D8")
    axes_s[0].fill_between(vals, areas_s, 0, where=areas_s>0, alpha=0.15, color="#CE93D8")
    axes_s[0].axhline(0, color="red", lw=1, ls="--", label="Open circuit")
    axes_s[0].set_ylabel("Area [nm²]"); axes_s[0].legend(fontsize=8); axes_s[0].grid(alpha=0.3)
    axes_s[0].set_title(f"{scan_p} → Junction Area")
    axes_s[1].plot(vals, ox_s, lw=2, color="#64B5F6", label="overlap x")
    axes_s[1].plot(vals, oy_s, lw=2, color="#EF9A9A", label="overlap y")
    axes_s[1].axhline(0, color="red", lw=1, ls="--")
    axes_s[1].set_ylabel("Overlap [nm]"); axes_s[1].legend(fontsize=8); axes_s[1].grid(alpha=0.3)
    axes_s[1].set_title(f"{scan_p} → Overlaps")
    axes_s[2].plot(vals, tilts_s, lw=2, color="#80CBC4")
    axes_s[2].axhline(0, color="#aaa", lw=0.8, ls=":")
    axes_s[2].set_ylabel("Tilt α [°]"); axes_s[2].set_xlabel(xlabel); axes_s[2].grid(alpha=0.3)
    axes_s[2].set_title(f"{scan_p} → Junction Tilt")
    fig_s.tight_layout()
    st.pyplot(fig_s, use_container_width=True)
    plt.close(fig_s)

# ═══ TAB 5: Junction Area ════════════════════════════════════════
with tab5:
    st.subheader("Junction Area & Full Parameter Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Overlap x (engine)",  f"{eng_ox:.0f} nm")
    c2.metric("Overlap y (engine)",  f"{eng_oy:.0f} nm")
    c3.metric("Area A (engine)",     f"{eng_area:.0f} nm²")
    c4.metric("Est. Ic (engine)",    f"{eng_ic:.3f} µA",
              help="Al, 4K: jc=10 kA/cm² (Ambegaokar-Baratoff)")
    st.caption(f"Engine voxel size = {eng.vox:.1f} nm "
               "(area/overlap resolution).")
    st.divider()
    if mode == "Dolan bridge":
        detail = {
            "Mode":                  params.mode,
            "PMMA [nm]":             params.t_pmma,
            "Bridge gap [nm]":       params.bridge_gap,
            "Total resist [nm]":     params.t_resist,
            "Undercut u [nm]":       params.undercut,
            "θ₁ [°]":                params.angle1,
            "φ₁ [°]":                params.phi1,
            "d₁ [nm]":               params.t_metal1,
            "sx₁ [nm]":              res["sx1"],
            "sy₁ [nm]":              res["sy1"],
            "θ₂ [°]":                params.angle2,
            "φ₂ [°]":                params.phi2,
            "d₂ [nm]":               params.t_metal2,
            "sx₂ [nm]":              res["sx2"],
            "sy₂ [nm]":              res["sy2"],
            "bridge_len [nm]":       params.bridge_len,
            "bridge_w [nm]":         params.bridge_w,
            "Overlap x (engine) [nm]":   eng_ox,
            "Overlap y (engine) [nm]":   eng_oy,
            "Junction area (engine) [nm²]": eng_area,
            "Estimated Ic (engine) [µA]":  eng_ic,
            "Engine voxel [nm]":     eng.vox,
        }
    else:
        detail = {
            "Mode":                  params.mode,
            "Deposition tilt θ [°]": params.manhattan_theta,
            "In-plane offset δ [°]": params.manhattan_delta,
            "Imaging resist h [nm]": params.manhattan_h,
            "x-arm opening wx [nm]": params.manhattan_wx,
            "y-arm opening wy [nm]": params.manhattan_wy,
            "φ₁ [°]":                params.phi1,
            "φ₂ [°]":                params.phi2,
            "shrink h·sinδ/tanθ [nm]": res.get("shrink_nm", 0.0),
            "Overlap x (engine) [nm]":   eng_ox,
            "Overlap y (engine) [nm]":   eng_oy,
            "Junction area (engine) [nm²]": eng_area,
            "Estimated Ic (engine) [µA]":  eng_ic,
            "Engine voxel [nm]":     eng.vox,
        }
    num_detail = {k:v for k,v in detail.items() if isinstance(v, (int,float))}
    str_detail = {k:v for k,v in detail.items() if isinstance(v, str)}
    rows = [(k, v) for k,v in str_detail.items()]
    rows += [(k, f"{v:.3f}") for k,v in num_detail.items()]
    st.dataframe(
        {"Parameter": [r[0] for r in rows],
         "Value":     [r[1] for r in rows]},
        use_container_width=True, hide_index=True)
    st.download_button("📋 Export JSON",
                       data=json.dumps({k: (v if isinstance(v,str) else float(v))
                                        for k,v in detail.items()}, indent=2),
                       file_name="jj_params.json", mime="application/json")
