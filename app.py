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
import altair as alt
import pandas as pd
import json, copy, os, sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

from gds_parser import load_gds_polygons, list_layers
from process_engine import (ProcessParams, shadow_vector, wafer_params,
                            evap_beams, wafer_local_angles)
from cross_section import draw_cross_section
from phi_cross_section import draw_junction_topview, draw_phi_scan
from manhattan_check import manhattan_break_check
from top_view import draw_top_view
from junction_area import compute_junction_area
from deposition3d import simulate, junction_footprint, junction_combos
import voxel_view as vv

# ─── Ray-scan resolution presets:  name → (max_cells_per_axis, min_voxel_nm) ──
RES_LEVELS = {
    "Standard (fast)":     (140, 6.0),
    "Fine":                (200, 4.0),
    "Ultra-fine (slow)":   (260, 3.0),
    "Extra-fine (slower)": (340, 2.5),
    "Maximum (slowest)":   (420, 2.0),
}

# ─── Wafer sizes (SEMI primary-flat specs):  label → (radius_mm, flat_chord_mm) ──
# The primary flat (オリフラ) chord sits at y = −d, d = sqrt(R² − (chord/2)²).
WAFER_SPECS = {
    "2 inch": (25.4, 15.88),
    "3 inch": (38.1, 22.22),
    "4 inch": (50.0, 32.50),
    "6 inch": (75.0, 57.50),
}

# Sidebar widget keys.  Mode-specific widgets get DISTINCT keys (prefixed) so
# switching modes can never feed an out-of-range value into a shared slider.
_SHARED_KEYS = ["t_pmma", "t_mma", "undercut", "angle1", "phi1", "t_metal1"]
_DOLAN_KEYS = {"angle2": "d_angle2", "phi2": "d_phi2", "t_metal2": "d_tmetal2",
               "bridge_len": "d_bridge_len", "bridge_w": "d_bridge_w",
               "bridge_pmma_gap": "d_bridge_pmma_gap"}
_MANH_KEYS = {"angle2": "m_angle2", "phi2": "m_phi2",
              "t_metal2": "m_tmetal2",
              "manhattan_wx": "m_wx", "manhattan_wy": "m_wy"}
# Trilayer sublayer params → widget keys (orthogonal to mode; applies to both).
_TRI_KEYS = {"tri_t1": "tri_t1", "tri_t2": "tri_t2",
             "tri_t3": "tri_t3", "tri_t4": "tri_t4",
             "tri_angle2": "tri_a2", "tri_phi2": "tri_p2",
             "tri_angle4": "tri_a4", "tri_phi4": "tri_p4"}
# (slider min, max) per widget key — used to clamp loaded values so an
# out-of-range file can never crash widget creation.
_KEY_RANGE = {
    "t_pmma": (100, 800), "t_mma": (100, 1500), "undercut": (0, 500),
    "angle1": (-60, 60), "phi1": (-90, 90), "t_metal1": (10, 200),
    "d_angle2": (-60, 60), "d_phi2": (-90, 90), "d_tmetal2": (10, 200),
    "d_bridge_len": (50, 2000), "d_bridge_w": (50, 1000),
    "d_bridge_pmma_gap": (0, 2000),
    "m_angle2": (-80, 80), "m_phi2": (-90, 180), "m_tmetal2": (10, 200),
    "m_wx": (100, 2000), "m_wy": (100, 2000),
    "tri_t1": (10, 300), "tri_t2": (1, 100), "tri_t3": (1, 100),
    "tri_t4": (10, 400), "tri_a2": (-80, 80), "tri_p2": (-90, 180),
    "tri_a4": (-80, 80), "tri_p4": (-90, 180),
}


def _set_clamped_int(key, val):
    lo, hi = _KEY_RANGE[key]
    st.session_state[key] = int(round(min(max(float(val), lo), hi)))


def _apply_loaded_params(pdict, raydict):
    """Push loaded parameter values into widget session_state (before widgets
    are created).  Returns the number of fields applied."""
    applied = 0
    mode_in = pdict.get("mode")
    if mode_in in ("Dolan bridge", "Manhattan"):
        st.session_state["mode"] = mode_in
        applied += 1
    stack_in = pdict.get("stack")
    if stack_in in ("Bilayer", "Trilayer"):
        st.session_state["stack"] = stack_in
        applied += 1
    for fld in _SHARED_KEYS:
        if pdict.get(fld) is not None:
            _set_clamped_int(fld, pdict[fld]); applied += 1
    keymap = (_DOLAN_KEYS if mode_in == "Dolan bridge"
              else _MANH_KEYS if mode_in == "Manhattan" else {})
    for fld, key in keymap.items():
        if pdict.get(fld) is not None:
            _set_clamped_int(key, pdict[fld]); applied += 1
    tri_loaded = False
    for fld, key in _TRI_KEYS.items():
        if pdict.get(fld) is not None:
            _set_clamped_int(key, pdict[fld]); applied += 1
            if fld in ("tri_angle2", "tri_phi2", "tri_angle4", "tri_phi4"):
                tri_loaded = True
    if tri_loaded:
        # Reproduce the saved evap-2 / evap-4 angles exactly: unlink them from
        # the primary tilt so the loaded θ/φ are used as-is.
        st.session_state["tri_link2"] = False
        st.session_state["tri_link4"] = False
    if raydict and raydict.get("resolution") in RES_LEVELS:
        st.session_state["res_level"] = raydict["resolution"]; applied += 1
    return applied


# ─── Josephson-junction electrical quantities ───────────────────────
_E_CHG = 1.602176634e-19         # elementary charge      [C]
_H_PL  = 6.62607015e-34          # Planck constant         [J·s]
_HBAR  = _H_PL / (2 * np.pi)     # reduced Planck constant [J·s]
_PHI0  = _H_PL / (2 * _E_CHG)    # magnetic flux quantum   [Wb]
_KB    = 1.380649e-23            # Boltzmann constant      [J/K]


def jj_electrical(ic_uA):
    """Josephson inductance & energy derived from the critical current Ic.

    L_J = ħ / (2e·Ic),   E_J = (Φ₀/2π)·Ic = ħ·Ic/2e.
    Returns L_J [nH], E_J [J], E_J/h [GHz] and E_J/kB [K].  For Ic ≤ 0
    (open circuit) L_J is infinite and E_J is zero."""
    ic = float(ic_uA) * 1e-6                       # A
    if ic <= 0:
        return dict(Lj_nH=float("inf"), Ej_J=0.0, Ej_h_GHz=0.0, Ej_kB_K=0.0)
    Lj = _HBAR / (2 * _E_CHG * ic)                 # H
    Ej = (_PHI0 / (2 * np.pi)) * ic                # J
    return dict(Lj_nH=Lj * 1e9, Ej_J=Ej,
                Ej_h_GHz=Ej / _H_PL / 1e9, Ej_kB_K=Ej / _KB)


def _fmt_lj(lj_nH):
    """Human-readable Josephson inductance (nH / µH, or ∞ for an open junction)."""
    if not np.isfinite(lj_nH):
        return "∞ (open)"
    if lj_nH >= 1000.0:
        return f"{lj_nH / 1000.0:.3f} µH"
    return f"{lj_nH:.3f} nH"


# Trilayer junction barrier composition (metal pair across the oxide).
_COMBO_ORDER = ["Nb-Al", "Al-Al", "Nb-Nb"]


def _combo_metrics(combos):
    """Show a 3-column row of per-pair junction areas (Nb-Al / Al-Al / Nb-Nb)."""
    st.markdown("**Junction barrier composition** — area by metal pair across the oxide")
    cols = st.columns(3)
    for col, name in zip(cols, _COMBO_ORDER):
        area = combos.get(name, {}).get("area", 0.0) if combos else 0.0
        col.metric(name, f"{area:.0f} nm²")


# Subscript digits for the per-evaporation angle labels (θ₁ … φ₄).
_SUB = {1: "₁", 2: "₂", 3: "₃", 4: "₄"}


def _tri_thickness(label, key, default, lo, hi, step):
    """Render one trilayer sublayer-thickness slider; return its value [nm]."""
    st.session_state.setdefault(key, default)
    return float(st.slider(label, lo, hi, step=step, key=key))


def _tri_linked_tilt(idx, src_idx, prim_angle, prim_phi,
                     a_key, p_key, a_range, p_range):
    """Tilt control for evaporation 2 / 4 (the second sublayer of an electrode).

    A "Same tilt as evap {src_idx}" checkbox (on by default) makes θ/φ follow
    the electrode's primary tilt; unchecking it reveals independent θ/φ sliders
    so each evaporation angle can be set freely.  Returns ``(angle, phi)``."""
    link_key = f"tri_link{idx}"
    st.session_state.setdefault(link_key, True)
    linked = st.checkbox(f"Same tilt as evap {src_idx}", key=link_key,
                         help=f"On → evap {idx} θ/φ follow evap {src_idx}.  "
                              f"Off → set evap {idx} θ/φ independently.")
    if linked:
        st.caption(f"θ{_SUB[idx]} = {prim_angle:.0f}°  ·  φ{_SUB[idx]} = "
                   f"{prim_phi:.0f}°   (following evap {src_idx})")
        return float(prim_angle), float(prim_phi)
    st.session_state.setdefault(a_key, int(round(prim_angle)))
    angle = st.slider(f"Polar θ{_SUB[idx]} [°]", *a_range, step=1, key=a_key)
    st.session_state.setdefault(p_key, int(round(prim_phi)))
    phi = st.slider(f"Azimuthal φ{_SUB[idx]} [°]", *p_range, step=1, key=p_key)
    return float(angle), float(phi)


# Default value of every process-parameter widget key (used by Reset button).
# Must match the per-widget `setdefault(...)` defaults in the sidebar below.
_PARAM_DEFAULTS = {
    "mode": "Dolan bridge",
    "stack": "Bilayer",
    "t_pmma": 250, "t_mma": 900, "undercut": 150,
    "angle1": -24, "phi1": 0, "t_metal1": 30,
    "d_angle2": 24, "d_phi2": 0, "d_tmetal2": 30,
    "d_bridge_len": 250, "d_bridge_w": 250, "d_bridge_pmma_gap": 0,
    "m_angle2": 60, "m_phi2": 90, "m_tmetal2": 30,
    "m_wx": 600, "m_wy": 600,
    "tri_t1": 80, "tri_t2": 10, "tri_t3": 10, "tri_t4": 150,
    "tri_a2": -24, "tri_p2": 0, "tri_a4": 24, "tri_p4": 0,
    "tri_link2": True, "tri_link4": True,
    "res_level": "Standard (fast)",
}


def _reset_defaults():
    """Restore every process-parameter widget to its default value."""
    for k, v in _PARAM_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("_imported_sig", None)   # allow re-loading later


def _build_export(params, eng, area, ox, oy, njunc, ic, juncs, res_level,
                  combos=None):
    """Serialise the full process (parameters + ray-scan + junction results)
    to a JSON string that can be re-loaded to restore every parameter."""
    juncs_out = [dict(area_nm2=float(j["area"]), overlap_x_nm=float(j["ox"]),
                      overlap_y_nm=float(j["oy"]), center_x_nm=float(j["cx"]),
                      center_y_nm=float(j["cy"]), cells=int(j["cells"]))
                 for j in juncs]
    _jj = jj_electrical(ic)
    _lj = _jj["Lj_nH"]
    results = {"junction_area_nm2": float(area),
               "overlap_x_nm": float(ox), "overlap_y_nm": float(oy),
               "n_junctions": int(njunc), "est_Ic_uA": float(ic),
               "L_J_nH": (None if not np.isfinite(_lj) else float(_lj)),
               "E_J_J": float(_jj["Ej_J"]),
               "E_J_over_h_GHz": float(_jj["Ej_h_GHz"]),
               "E_J_over_kB_K": float(_jj["Ej_kB_K"]),
               "junctions": juncs_out}
    if combos:
        results["barrier_composition_nm2"] = {
            name: float(d["area"]) for name, d in combos.items()}
    out = {
        "shadowcast": "v6",
        "mode": params.mode,
        "stack": params.stack,
        "parameters": asdict(params),
        "ray_scan": {"resolution": res_level,
                     "max_cells": int(eng.meta.get("max_cells", 0)),
                     "min_vox_nm": float(eng.meta.get("min_vox", 0.0)),
                     "voxel_nm": float(eng.vox)},
        "results": results,
    }
    return json.dumps(out, indent=2)


st.set_page_config(page_title="ShadowCast", page_icon="⚛️", layout="wide")
st.title("⚛️ ShadowCast — Josephson Junction Process Simulator")
st.caption(
    "Bilayer resist (PMMA/MMA) · θ & φ evaporation · Sidewall shadowing · "
    "Oxidation step · **Dolan bridge** (bridge_len × bridge_w, side trenches open) · "
    "**Manhattan cross** (φ₁ ⊥ φ₂)"
)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    # ── Save / Load ───────────────────────────────────────────────
    st.header("💾 Save / Load")
    _imp = st.file_uploader("Load parameters (JSON)", type=["json"],
                            key="param_import")
    if _imp is not None:
        _sig = (_imp.name, _imp.size)
        if st.session_state.get("_imported_sig") != _sig:
            try:
                _data = json.load(_imp)
                _pdict = _data.get("parameters", _data)
                _raydict = _data.get("ray_scan", {})
                _n = _apply_loaded_params(_pdict, _raydict)
                st.session_state["_imported_sig"] = _sig
                st.success(f"✅ Loaded {_n} parameters — applying…")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")
    # Filled with the export download button after the engine has run.
    _save_box = st.container()
    if st.button("↺ Reset parameters to defaults", use_container_width=True,
                 help="Restore every process parameter (and resolution) to its "
                      "default value."):
        _reset_defaults()
        st.rerun()
    st.divider()

    st.header("📂 GDS File")
    uploaded = st.file_uploader("Upload GDS / GDSII", type=["gds", "gdsii"])

    st.header("🔧 Process Parameters")
    st.session_state.setdefault("mode", "Dolan bridge")
    mode = st.radio("Junction type", ["Dolan bridge", "Manhattan"],
                    horizontal=True, key="mode")
    st.session_state.setdefault("stack", "Bilayer")
    stack = st.radio(
        "Deposition stack", ["Bilayer", "Trilayer"], horizontal=True,
        key="stack",
        help="Bilayer: evap 1 → oxidation → evap 2.  "
             "Trilayer: Nb→Al → oxidation → Al→Nb (Nb/Al/Al/Nb), forming a "
             "junction whose barrier metals are reported as Nb-Al / Al-Al / Nb-Nb.")

    # Trilayer sublayer values.  The engine reads these only when
    # stack == "Trilayer"; in that case the per-evaporation blocks below set
    # them.  Bilayer leaves them at these inert defaults.
    tri_t1, tri_t2, tri_t3, tri_t4 = 80.0, 10.0, 10.0, 150.0
    tri_angle2 = tri_phi2 = tri_angle4 = tri_phi4 = 0.0

    st.subheader("Bilayer resist")
    st.caption("Recipe arxiv:2101.01453 — PMMA A-4 ≈250 nm / MMA EL-13 ≈900 nm")
    st.session_state.setdefault("t_pmma", 250)
    t_pmma = st.slider("PMMA [nm]  (top, no undercut)", 100, 800,
                       step=25, key="t_pmma")
    st.session_state.setdefault("t_mma", 900)
    t_mma = st.slider("MMA [nm]  (bottom = bridge height / vertical gap)",
                      100, 1500, step=25, key="t_mma",
                      help="MMA bottom-layer thickness.  The bridge underside "
                           "sits at z = MMA height, so this sets the vertical "
                           "shadow gap.  Junction overlap ≈ 2·MMA·tanθ − bridge width.")
    st.session_state.setdefault("undercut", 150)
    undercut = st.slider("MMA undercut u [nm]  (one-sided)", 0, 500,
                         step=10, key="undercut")
    st.caption(f"Total resist: {t_pmma+t_mma} nm  ·  vertical shadow gap = MMA = {t_mma} nm")

    st.subheader("Evaporation 1" + (" — Nb" if stack == "Trilayer" else ""))
    if stack == "Trilayer":
        st.caption("Electrode-1 primary tilt θ₁/φ₁ (drives evap 1 Nb; evap 2 Al "
                   "follows it unless you unlink it below).")
    st.session_state.setdefault("angle1", -24)
    angle1 = st.slider("Polar θ₁ [°]", -60, 60, step=1, key="angle1")
    st.session_state.setdefault("phi1", 0)
    phi1 = st.slider("Azimuthal φ₁ [°]", -90, 90, step=1, key="phi1")
    st.session_state.setdefault("t_metal1", 30)
    if stack == "Bilayer":
        t_metal1 = st.slider("Metal d₁ [nm]", 10, 200, step=5, key="t_metal1")
    else:
        t_metal1 = float(st.session_state["t_metal1"])  # unused by trilayer engine
        tri_t1 = _tri_thickness("Nb d₁ [nm]  (evap 1)", "tri_t1", 80, 10, 300, 5)

        st.subheader("Evaporation 2 — Al")
        tri_t2 = _tri_thickness("Al d₂ [nm]  (evap 2)", "tri_t2", 10, 1, 100, 1)
        tri_angle2, tri_phi2 = _tri_linked_tilt(
            2, 1, angle1, phi1, "tri_a2", "tri_p2", (-80, 80), (-90, 180))

    if mode == "Dolan bridge":
        _tri = stack == "Trilayer"
        st.subheader("Evaporation 3 — Al" if _tri else "Evaporation 2")
        if _tri:
            st.caption("Electrode-2 primary tilt θ₃/φ₃ (drives evap 3 Al; evap 4 "
                       "Nb follows it unless you unlink it below).")
        st.session_state.setdefault("d_angle2", 24)
        angle2 = st.slider(f"Polar θ{'₃' if _tri else '₂'} [°]", -60, 60, step=1,
                           key="d_angle2",
                           help=("Electrode-2 tilt." if _tri else
                                 "Dolan = uniaxial tilt: φ₂=φ₁, θ₂=−θ₁"))
        st.session_state.setdefault("d_phi2", 0)
        phi2 = st.slider(f"Azimuthal φ{'₃' if _tri else '₂'} [°]", -90, 90,
                         step=1, key="d_phi2")
        st.session_state.setdefault("d_tmetal2", 30)
        if not _tri:
            t_metal2 = st.slider("Metal d₂ [nm]", 10, 200, step=5, key="d_tmetal2")
        else:
            t_metal2 = float(st.session_state["d_tmetal2"])  # unused by trilayer
            tri_t3 = _tri_thickness("Al d₃ [nm]  (evap 3)", "tri_t3", 10, 1, 100, 1)
            st.subheader("Evaporation 4 — Nb")
            tri_t4 = _tri_thickness("Nb d₄ [nm]  (evap 4)", "tri_t4", 150, 10, 400, 5)
            tri_angle4, tri_phi4 = _tri_linked_tilt(
                4, 3, angle2, phi2, "tri_a4", "tri_p4", (-80, 80), (-90, 180))

        st.subheader("Geometry")
        st.markdown("Bridge slab dimensions:")
        st.session_state.setdefault("d_bridge_len", 250)
        bridge_len = st.slider("Bridge width [nm]  (evap direction x, shadow-defining)",
                               50, 2000, step=10, key="d_bridge_len")
        st.session_state.setdefault("d_bridge_w", 250)
        bridge_w = st.slider("Bridge length [nm]  (junction width, y)",
                             50, 1000, step=10, key="d_bridge_w")
        st.session_state.setdefault("d_bridge_pmma_gap", 0)
        bridge_pmma_gap = st.slider(
            "Bridge ↔ PMMA opening [nm]  (per side, 0 = auto)",
            0, 2000, step=25, key="d_bridge_pmma_gap",
            help="Horizontal gap between each bridge edge and the PMMA wall "
                 "(the trench window). 0 auto-sizes it wide enough for the "
                 "tilted beam to reach under the bridge.")
        t_shadow = t_mma * np.tan(np.radians(max(abs(angle1), abs(angle2))))
        st.caption(
            f"Shadow projection = MMA · tan θ = {t_shadow:.0f} nm  \n"
            f"Junction when bridge width **<** {2*t_shadow:.0f} nm  \n"
            f"(bridge_len < 2 · MMA · tan θ  →  deposits meet under bridge)"
        )
        manhattan_wx = manhattan_wy = bridge_w
        manhattan_theta = 60.0; manhattan_delta = 15.0; manhattan_h = 1800.0
    else:
        st.caption("Recipe arxiv:2605.19590 — two oblique beams (θ ≈ 60°), "
                   "azimuths ≈ 90° apart (Manhattan crossing).")
        # Evaporation 1 (θ₁, φ₁, d₁) comes from the shared section above.
        _tri = stack == "Trilayer"
        st.subheader("Evaporation 3 — Al" if _tri else "Evaporation 2")
        if _tri:
            st.caption("Electrode-2 primary tilt θ₃/φ₃ (drives evap 3 Al; evap 4 "
                       "Nb follows it unless you unlink it below).")
        st.session_state.setdefault("m_angle2", 60)
        angle2 = st.slider(f"Polar θ{'₃' if _tri else '₂'} [°]", -80, 80, step=1,
                           key="m_angle2",
                           help="Tilt of the second beam from the surface normal.")
        st.session_state.setdefault("m_phi2", 90)
        phi2 = st.slider(f"Azimuthal φ{'₃' if _tri else '₂'} [°]", -90, 180,
                         step=1, key="m_phi2",
                         help="Default 90° → perpendicular to Evap 1")
        st.session_state.setdefault("m_tmetal2", 30)
        if not _tri:
            t_metal2 = st.slider("Metal d₂ [nm]", 10, 200, step=5, key="m_tmetal2")
        else:
            t_metal2 = float(st.session_state["m_tmetal2"])  # unused by trilayer
            tri_t3 = _tri_thickness("Al d₃ [nm]  (evap 3)", "tri_t3", 10, 1, 100, 1)
            st.subheader("Evaporation 4 — Nb")
            tri_t4 = _tri_thickness("Nb d₄ [nm]  (evap 4)", "tri_t4", 150, 10, 400, 5)
            tri_angle4, tri_phi4 = _tri_linked_tilt(
                4, 3, angle2, phi2, "tri_a4", "tri_p4", (-80, 80), (-90, 180))

        st.subheader("Geometry")
        st.markdown("Designed resist line openings (Manhattan crossing):")
        st.session_state.setdefault("m_wx", 600)
        manhattan_wx = st.slider("x-arm opening wx [nm]", 100, 2000,
                                 step=10, key="m_wx")
        st.session_state.setdefault("m_wy", 600)
        manhattan_wy = st.slider("y-arm opening wy [nm]", 100, 2000,
                                 step=10, key="m_wy")
        bridge_len = 700; bridge_w = 300; bridge_pmma_gap = 0.0
        # Derived quantities for the analytic estimate / status messages only
        # (the 3D engine is the source of truth and uses θ₁/φ₁/θ₂/φ₂ directly).
        manhattan_theta = (abs(angle1) + abs(angle2)) / 2.0
        manhattan_delta = max(abs(phi1), abs(phi2 - 90.0), 1.0)
        manhattan_h = float(t_pmma + t_mma)

    st.divider()
    st.subheader("Ray-scan resolution")
    st.session_state.setdefault("res_level", "Standard (fast)")
    res_level = st.selectbox(
        "Voxel grid density", list(RES_LEVELS.keys()), key="res_level",
        help="Finer = the tilted beam is ray-traced into a denser voxel grid "
             "(smaller voxels) for sharper metal / junction edges, at the cost "
             "of speed and memory.")
    _max_cells, _min_vox = RES_LEVELS[res_level]

    st.divider()
    st.subheader("Display")
    show_shadow   = st.checkbox("Show shadow deposits (top view)", True)
    show_undercut = st.checkbox("Show undercut regions (top view)", True)

params = ProcessParams(
    t_pmma=t_pmma, t_mma=t_mma, undercut=undercut,
    angle1=angle1, phi1=phi1, t_metal1=t_metal1,
    angle2=angle2, phi2=phi2, t_metal2=t_metal2,
    bridge_len=bridge_len, bridge_w=bridge_w, bridge_pmma_gap=bridge_pmma_gap,
    manhattan_wx=manhattan_wx, manhattan_wy=manhattan_wy,
    manhattan_theta=manhattan_theta, manhattan_delta=manhattan_delta,
    manhattan_h=manhattan_h,
    mode=mode,
    stack=stack,
    tri_t1=tri_t1, tri_t2=tri_t2, tri_t3=tri_t3, tri_t4=tri_t4,
    tri_angle2=tri_angle2, tri_phi2=tri_phi2,
    tri_angle4=tri_angle4, tri_phi4=tri_phi4,
)
res = compute_junction_area(params)

# ─── 3D physical deposition engine (source of truth) ──────────────
@st.cache_data(show_spinner=False)
def _run_engine(ekey, _params, max_cells, min_vox):
    r = simulate(_params, max_cells=max_cells, min_vox=min_vox)
    jm, area, ox, oy, juncs = junction_footprint(r)
    combos, combo_map = junction_combos(r)
    return r, jm, area, ox, oy, juncs, combos, combo_map

ekey = (params.mode, params.t_pmma, params.t_mma, params.undercut,
        params.angle1, params.phi1, params.t_metal1,
        params.angle2, params.phi2, params.t_metal2,
        params.bridge_len, params.bridge_w, params.bridge_pmma_gap,
        params.manhattan_wx, params.manhattan_wy,
        params.manhattan_theta, params.manhattan_delta, params.manhattan_h,
        _max_cells, _min_vox,
        params.stack, params.tri_t1, params.tri_t2, params.tri_t3, params.tri_t4,
        params.tri_angle2, params.tri_phi2, params.tri_angle4, params.tri_phi4)
with st.spinner("Running 3D shadow-evaporation engine..."):
    (eng, eng_jm, eng_area, eng_ox, eng_oy, eng_juncs,
     eng_combos, eng_combo_map) = _run_engine(
        ekey, params, _max_cells, _min_vox)
eng_njunc = len(eng_juncs)
# Engine-based critical current (jc = 10 kA/cm², Ambegaokar-Baratoff),
# consistent with junction_area.py:  Ic[µA] = area_nm2 · 1e-4
eng_ic = eng_area * 1e-4
# Josephson inductance L_J and energy E_J derived from that critical current.
eng_jj = jj_electrical(eng_ic)

# Now the engine has run, fill the sidebar Save box with a download button that
# exports every parameter + the junction results (re-loadable via the uploader).
with _save_box:
    st.download_button(
        "💾 Save parameters + results",
        data=_build_export(params, eng, eng_area, eng_ox, eng_oy,
                           eng_njunc, eng_ic, eng_juncs, res_level,
                           combos=eng_combos),
        file_name="shadowcast_params.json", mime="application/json",
        use_container_width=True)

# ─── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab_scan, tab_wafer, tab5 = st.tabs([
    "📐 Cross-section",
    "🗺️ Top View",
    "🔄 φ Junction View",
    "🔍 Break Check",
    "📈 Parameter Scan",
    "🌐 Wafer Map",
    "📊 Junction Area",
])

# ═══ TAB 1: Cross-section ════════════════════════════════════════
with tab1:
    st.subheader("Cross-section  (3D voxel engine — rotatable slice)")
    st.markdown(
        "Metal positions are computed by the **physical shadow-evaporation engine**: "
        "the real 3-D resist geometry is built as boxes and the tilted beam is "
        "ray-traced into a voxel grid.  The slice can be **rotated** to any "
        "in-plane angle and shifted perpendicular to itself."
    )

    _gR = float(eng.meta.get("grid_R", eng.meta["R"]))

    # Slice orientation: azimuth angle α (0° = x–z cut, 90° = y–z cut) and a
    # perpendicular offset.  Persist both across reruns (key-only sliders).
    st.session_state.setdefault("slice_angle", 0.0)
    st.session_state["slice_angle"] = float(
        np.clip(st.session_state["slice_angle"], -90.0, 90.0))
    st.session_state.setdefault("slice_off", 0.0)
    st.session_state["slice_off"] = float(
        np.clip(st.session_state["slice_off"], -_gR, _gR))
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        slice_angle = st.slider("Slice angle α [°]  (0 = x–z, ±90 = y–z)",
                                -90.0, 90.0, step=1.0, key="slice_angle")
    with cc2:
        slice_pos = st.slider("Perpendicular offset [nm]", -_gR, _gR,
                              step=eng.vox, key="slice_off")

    _ztop = float(eng.zs[-1])
    # Persist the view range across reruns (e.g. moving the slice slider): seed
    # once, then clamp the stored value to the current grid extent.  The slider
    # is created with `key` only (no `value=`) so it never snaps back to a
    # default when something else on the page triggers a rerun.
    st.session_state.setdefault("cs_half", min(_gR, 800.0))
    st.session_state.setdefault("cs_zmax", min(_ztop, 600.0))
    st.session_state["cs_half"] = float(np.clip(st.session_state["cs_half"], 100.0, _gR))
    st.session_state["cs_zmax"] = float(np.clip(st.session_state["cs_zmax"], 100.0, _ztop))
    with st.expander("🔍 表示範囲 (View range)", expanded=False):
        vr1, vr2 = st.columns(2)
        cs_half = vr1.slider("Horizontal half-width [nm]", 100.0, _gR,
                             step=50.0, key="cs_half")
        cs_zmax = vr2.slider("Z max [nm]", 100.0, _ztop,
                             step=50.0, key="cs_zmax")

    # Orientation aid: show WHERE the chosen slice cuts through the device on a
    # top view (dashed yellow line), so the cross section is easy to locate.
    lc1, lc2 = st.columns([1, 1])
    with lc1:
        st.markdown("**Slice location** (top view)")
        with st.spinner("Locating slice..."):
            figloc = vv.render_top_view(eng, eng_jm, view_half=cs_half,
                                        juncs=eng_juncs,
                                        slice_line=(slice_angle, slice_pos),
                                        combo_map=eng_combo_map)
            st.pyplot(figloc, use_container_width=True)
            plt.close(figloc)

    with st.spinner("Slicing voxel grid..."):
        st.markdown("**Process stages** — resist → evap 1 → oxidation → evap 2 → lift-off")
        figs = vv.render_stages(eng, slice_angle, slice_pos, eng_jm,
                                view_half=cs_half, zmax=cs_zmax)
        st.pyplot(figs, use_container_width=True)
        plt.close(figs)
        st.markdown("**Combined slice** (all layers, junction highlighted)")
        figc = vv.render_cross_section(eng, slice_angle, slice_pos, eng_jm,
                                       view_half=cs_half, zmax=cs_zmax)
        st.pyplot(figc, use_container_width=True)
        plt.close(figc)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voxel size", f"{eng.vox:.1f} nm")
    c2.metric("Junction area (engine)", f"{eng_area:.0f} nm²")
    c3.metric("Overlap x (engine)", f"{eng_ox:.0f} nm")
    c4.metric("Overlap y (engine)", f"{eng_oy:.0f} nm")
    st.caption(
        "Junction area = the **true Al1∩Al2 overlap** measured by counting "
        "overlapping voxels (Σ cells × voxel²), so non-rectangular junctions "
        "are handled exactly; overlap x / y are just the bounding-box extents."
    )
    e1, e2, e3 = st.columns(3)
    e1.metric("Est. critical current Iᶜ", f"{eng_ic:.3f} µA",
              help="Al, ~4 K: jc = 10 kA/cm² (Ambegaokar–Baratoff)")
    e2.metric("Josephson inductance L_J", _fmt_lj(eng_jj["Lj_nH"]),
              help="L_J = ħ / (2e·Iᶜ)")
    e3.metric("Josephson energy E_J/h", f"{eng_jj['Ej_h_GHz']:.2f} GHz",
              help=f"E_J = (Φ₀/2π)·Iᶜ = {eng_jj['Ej_kB_K']:.2f} K·k_B")

    if params.stack == "Trilayer":
        _combo_metrics(eng_combos)

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
    elif eng_njunc >= 2:
        st.success(f"✅ {eng_njunc} junctions formed "
                   f"(largest {eng_ox:.0f} × {eng_oy:.0f} nm; total area "
                   f"{eng_area:.0f} nm²). See Top View tab for J1…J{eng_njunc}.")
    else:
        st.success(f"✅ Junction formed: {eng_ox:.0f} × {eng_oy:.0f} nm (engine)")

    if params.stack == "Trilayer":
        st.info(
            "**Colour guide (trilayer):**  grey = substrate · tan = resist · "
            "amber = Nb · light-blue = Al · purple = AlOx.  Junction barrier: "
            "gold = Nb-Al · teal = Al-Al · raspberry = Nb-Nb."
        )
    else:
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
    st.session_state.setdefault("top_half", min(_gR2, 1000.0))
    st.session_state["top_half"] = float(np.clip(st.session_state["top_half"], 100.0, _gR2))
    with st.expander("🔍 表示範囲 (View range)", expanded=False):
        top_half = st.slider("Half-width [nm]", 100.0, _gR2,
                             step=50.0, key="top_half")

    with st.spinner("Rendering staged top view..."):
        figts = vv.render_top_stages(eng, eng_jm, view_half=top_half,
                                     juncs=eng_juncs, combo_map=eng_combo_map)
        st.pyplot(figts, use_container_width=True)
        plt.close(figts)

    st.markdown("**Final floor deposit** (combined)")
    with st.spinner("Rendering floor map..."):
        figt = vv.render_top_view(eng, eng_jm, view_half=top_half,
                                  juncs=eng_juncs, combo_map=eng_combo_map)
        st.pyplot(figt, use_container_width=True)
        plt.close(figt)

    st.divider()
    st.markdown("**Lift-off film thickness** — z value = stacked metal thickness "
                "(electrode overlap = thicker)")
    tc1, tc2 = st.columns(2)
    with tc1:
        with st.spinner("Rendering thickness heat map..."):
            figth = vv.render_thickness_map(eng, view_half=top_half)
            st.pyplot(figth, use_container_width=True)
            plt.close(figth)
    with tc2:
        with st.spinner("Rendering 3D thickness surface..."):
            figth3 = vv.render_thickness_surface(eng, view_half=top_half)
            st.pyplot(figth3, use_container_width=True)
            plt.close(figth3)

    if eng_njunc >= 2:
        st.warning(f"⚠️ {eng_njunc} separate Josephson junctions detected "
                   "(labelled J1, J2, … above, largest first).")
        st.dataframe({
            "Junction":      [f"J{i}" for i in range(1, eng_njunc + 1)],
            "Area [nm²]":    [round(j["area"]) for j in eng_juncs],
            "Overlap x [nm]":[round(j["ox"]) for j in eng_juncs],
            "Overlap y [nm]":[round(j["oy"]) for j in eng_juncs],
            "Center x [nm]": [round(j["cx"]) for j in eng_juncs],
            "Center y [nm]": [round(j["cy"]) for j in eng_juncs],
        }, use_container_width=True, hide_index=True)
    elif eng_njunc == 1:
        st.caption("Single Josephson junction.")

# ═══ TAB 3: φ Junction View ══════════════════════════════════════
with tab3:
    st.subheader("Junction Map (engine) + φ azimuth view")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Engine junction map** (floor deposit, JJ highlighted)")
        with st.spinner("Rendering..."):
            fig3 = vv.render_top_view(eng, eng_jm, juncs=eng_juncs,
                                      combo_map=eng_combo_map)
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
    d1, d2, d3 = st.columns(3)
    d1.metric("Josephson inductance L_J", _fmt_lj(eng_jj["Lj_nH"]),
              help="L_J = ħ / (2e·Iᶜ)")
    d2.metric("Josephson energy E_J/h", f"{eng_jj['Ej_h_GHz']:.2f} GHz",
              help="E_J = (Φ₀/2π)·Iᶜ")
    d3.metric("E_J / k_B", f"{eng_jj['Ej_kB_K']:.2f} K")

    if eng_area <= 0:
        if mode == "Dolan bridge":
            t_shadow_total = 2 * params.t_mma * np.tan(
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

# ═══ TAB: Parameter Scan ═════════════════════════════════════════
with tab_scan:
    st.subheader("Parameter Scan")
    st.caption("Sweeps the **3D engine** (source of truth) over one or two "
               "parameters. Pick variables, then **Run scan** — every metric "
               "is plotted.")

    # Per-mode scan variables: label → (param attr, lo, hi, axis label)
    if mode == "Dolan bridge":
        scan_vars = {
            "θ₁":            ("angle1",          -55,  55,  "θ₁ [°]"),
            "θ₂":            ("angle2",          -55,  55,  "θ₂ [°]"),
            "φ₁":            ("phi1",            -90,  90,  "φ₁ [°]"),
            "φ₂":            ("phi2",            -90,  90,  "φ₂ [°]"),
            "bridge_len":    ("bridge_len",      100,  2000,"bridge_len [nm]"),
            "bridge_w":      ("bridge_w",        50,   800, "bridge_w [nm]"),
            "MMA height":    ("t_mma",           100,  1500,"MMA height [nm]"),
            "Bridge↔PMMA gap":("bridge_pmma_gap", 0,   2000,"bridge↔PMMA gap [nm]"),
            "Undercut u":    ("undercut",        0,    400, "u [nm]"),
            "PMMA thickness":("t_pmma",          100,  800, "PMMA [nm]"),
        }
    else:
        scan_vars = {
            "θ₁":         ("angle1",       -80, 80,  "θ₁ [°]"),
            "φ₁":         ("phi1",         -90, 90,  "φ₁ [°]"),
            "θ₂":         ("angle2",       -80, 80,  "θ₂ [°]"),
            "φ₂":         ("phi2",         -90, 180, "φ₂ [°]"),
            "x-arm wx":   ("manhattan_wx", 100, 2000,"wx [nm]"),
            "y-arm wy":   ("manhattan_wy", 100, 2000,"wy [nm]"),
            "MMA height": ("t_mma",        100, 1500,"MMA height [nm]"),
            "Undercut u": ("undercut",     0,   400, "u [nm]"),
        }
    var_names = list(scan_vars.keys())

    # Every metric is plotted (stacked); colour per metric for the plots.
    metric_opts = ["Junction area [nm²]", "Est. Ic [µA]",
                   "L_J [nH]", "E_J/h [GHz]", "E_J/k_B [K]"]
    _metric_color = {"Junction area [nm²]": "#CE93D8", "Est. Ic [µA]": "#64B5F6",
                     "L_J [nH]": "#80CBC4", "E_J/h [GHz]": "#FFB74D",
                     "E_J/k_B [K]": "#EF9A9A"}

    def _metric_from_area(area_nm2, metric):
        ic = area_nm2 * 1e-4
        jj = jj_electrical(ic)
        if metric == "Junction area [nm²]": return area_nm2
        if metric == "Est. Ic [µA]":        return ic
        if metric == "L_J [nH]":
            return jj["Lj_nH"] if np.isfinite(jj["Lj_nH"]) else np.nan
        if metric == "E_J/h [GHz]":         return jj["Ej_h_GHz"]
        return jj["Ej_kB_K"]

    cdim, cres = st.columns(2)
    scan_dim = cdim.radio("Scan type", ["1D", "2D"], horizontal=True)
    scan_res = cres.selectbox("Voxel grid density", list(RES_LEVELS.keys()),
                              help="Same options as the sidebar ray-scan "
                                   "resolution (finer = slower).")
    _smc, _smv = RES_LEVELS[scan_res]

    def _scan_sig(p):
        return (p.mode, p.t_pmma, p.t_mma, p.undercut, p.angle1, p.phi1,
                p.t_metal1, p.angle2, p.phi2, p.t_metal2, p.bridge_len,
                p.bridge_w, p.bridge_pmma_gap, p.manhattan_wx, p.manhattan_wy,
                p.manhattan_theta, p.manhattan_delta, p.manhattan_h, _smc, _smv,
                p.stack, p.tri_t1, p.tri_t2, p.tri_t3, p.tri_t4,
                p.tri_angle2, p.tri_phi2, p.tri_angle4, p.tri_phi4)

    @st.cache_data(show_spinner=False)
    def _scan_area(sig, _p):
        r = simulate(_p, max_cells=_smc, min_vox=_smv)
        _, area, _, _, _ = junction_footprint(r)
        return float(area)

    if scan_dim == "1D":
        xv = st.selectbox("Scan variable", var_names, key="scan_x1d")
        xattr, xlo, xhi, xlabel = scan_vars[xv]
        cx, cy, cn = st.columns(3)
        # Range/step keyed per variable → each remembers its own bounds.
        vmin = cx.number_input("Min", value=float(xlo), step=1.0,
                               key=f"s1lo_{xv}")
        vmax = cy.number_input("Max", value=float(xhi), step=1.0,
                               key=f"s1hi_{xv}")
        npts = cn.number_input("Points", min_value=2, max_value=201, value=31,
                               step=1, key="scan_n1d")
        if st.button("▶ Run scan", key="run_scan_1d", use_container_width=True):
            if vmax <= vmin:
                st.warning("Max must be greater than Min.")
            else:
                xs = np.linspace(vmin, vmax, int(npts))
                areas = np.zeros(len(xs))
                prog = st.progress(0.0, text="Scanning…")
                for i, vx in enumerate(xs):
                    p2 = copy.copy(params); setattr(p2, xattr, float(vx))
                    areas[i] = _scan_area(_scan_sig(p2), p2)
                    prog.progress((i + 1) / len(xs),
                                  text=f"Scanning… {i+1}/{len(xs)}")
                prog.empty()
                st.session_state["_scan1d"] = dict(
                    xs=xs, areas=areas, xattr=xattr, xlabel=xlabel, var=xv)
        sc = st.session_state.get("_scan1d")
        if sc:
            n = len(metric_opts)
            fig_s, axes_s = plt.subplots(n, 1, figsize=(8, 2.6 * n),
                                         sharex=False)
            for ax, metric in zip(np.atleast_1d(axes_s), metric_opts):
                yvals = np.array([_metric_from_area(a, metric)
                                  for a in sc["areas"]])
                ax.plot(sc["xs"], yvals, lw=2.2, marker="o", ms=3,
                        color=_metric_color[metric])
                if metric in ("Junction area [nm²]", "Est. Ic [µA]"):
                    ax.axhline(0, color="red", lw=1, ls="--", label="Open circuit")
                    ax.legend(fontsize=8)
                ax.set_xlabel(sc["xlabel"])   # per-plot x label (not shared)
                ax.set_ylabel(metric)
                ax.set_title(f"{sc['var']} → {metric}")
                ax.grid(alpha=0.3)
            fig_s.suptitle(f"Engine scan ({scan_res})", y=1.0, fontsize=10)
            fig_s.tight_layout()
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)
    else:
        cxx, cyy = st.columns(2)
        xv = cxx.selectbox("X variable", var_names, key="scan_x2d")
        yv = cyy.selectbox("Y variable", var_names,
                           index=min(1, len(var_names) - 1), key="scan_y2d")
        xattr, xlo, xhi, xlabel = scan_vars[xv]
        yattr, ylo, yhi, ylabel = scan_vars[yv]
        rx1, rx2, rxn = st.columns(3)
        xmin = rx1.number_input("X min", value=float(xlo), step=1.0, key=f"s2xlo_{xv}")
        xmax = rx2.number_input("X max", value=float(xhi), step=1.0, key=f"s2xhi_{xv}")
        nx = rxn.number_input("X points", min_value=2, max_value=81, value=15,
                              step=1, key="scan_nx2d")
        ry1, ry2, ryn = st.columns(3)
        ymin = ry1.number_input("Y min", value=float(ylo), step=1.0, key=f"s2ylo_{yv}")
        ymax = ry2.number_input("Y max", value=float(yhi), step=1.0, key=f"s2yhi_{yv}")
        ny = ryn.number_input("Y points", min_value=2, max_value=81, value=15,
                              step=1, key="scan_ny2d")
        st.caption(f"{int(nx)}×{int(ny)} = {int(nx)*int(ny)} engine runs")
        if xv == yv:
            st.warning("Choose two different variables for a 2D scan.")
        elif xmax <= xmin or ymax <= ymin:
            st.warning("Each Max must be greater than its Min.")
        elif st.button("▶ Run scan", key="run_scan_2d", use_container_width=True):
            xs = np.linspace(xmin, xmax, int(nx))
            ys = np.linspace(ymin, ymax, int(ny))
            areas = np.zeros((len(ys), len(xs)))
            total = len(xs) * len(ys); done = 0
            prog = st.progress(0.0, text="Scanning…")
            for iy, vy in enumerate(ys):
                for ix, vx in enumerate(xs):
                    p2 = copy.copy(params)
                    setattr(p2, xattr, float(vx)); setattr(p2, yattr, float(vy))
                    areas[iy, ix] = _scan_area(_scan_sig(p2), p2)
                    done += 1
                    prog.progress(done / total, text=f"Scanning… {done}/{total}")
            prog.empty()
            st.session_state["_scan2d"] = dict(
                xs=xs, ys=ys, areas=areas, xlabel=xlabel, ylabel=ylabel,
                xvar=xv, yvar=yv)
        sc = st.session_state.get("_scan2d")
        if sc:
            n = len(metric_opts)
            fig_s, axes_s = plt.subplots(n, 1, figsize=(7.0, 5.2 * n))
            for ax, metric in zip(np.atleast_1d(axes_s), metric_opts):
                zz = np.vectorize(lambda a: _metric_from_area(a, metric))(sc["areas"])
                pc = ax.pcolormesh(sc["xs"], sc["ys"], zz, shading="auto",
                                   cmap="viridis")
                fig_s.colorbar(pc, ax=ax, label=metric)
                ax.set_xlabel(sc["xlabel"]); ax.set_ylabel(sc["ylabel"])
                ax.set_title(f"{sc['yvar']} vs {sc['xvar']} → {metric}")
            fig_s.suptitle(f"Engine scan ({scan_res})", y=1.0, fontsize=10)
            fig_s.tight_layout()
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)

# ═══ TAB: Wafer Map (Plassys point-source / tilted-wafer) ════════
with tab_wafer:
    st.subheader("Wafer Map — JJ-area variation across the wafer")
    st.caption(
        "Plassys-style oblique evaporation: a **fixed point source** with the "
        "**wafer tilted** to the nominal (θ, φ) at the wafer centre. Because the "
        "source is at a finite throw distance, an off-centre device sees a "
        "slightly different local angle, so the **junction area drifts with "
        "wafer position**. The device is replicated across a real wafer disk "
        "(with its primary flat / オリフラ); the centre reproduces the "
        "single-JJ result.")

    wc1, wc2, wc3, wc4 = st.columns(4)
    waf_L = wc1.number_input(
        "Throw distance L [mm]", min_value=20.0, max_value=2000.0,
        value=200.0, step=10.0, key="waf_L",
        help="Source→wafer-centre distance. Smaller L ⇒ stronger position "
             "dependence (deviation ≈ r/L).")
    waf_size = wc2.selectbox(
        "Wafer size", list(WAFER_SPECS.keys()), index=2, key="waf_size",
        help="Real wafer disk with its primary flat (オリフラ) at the bottom; "
             "the grid spans the whole wafer and off-wafer cells are skipped.")
    R_mm, c_flat = WAFER_SPECS[waf_size]
    d_flat = float(np.sqrt(R_mm ** 2 - (c_flat / 2.0) ** 2))
    waf_n = int(wc3.number_input(
        "Grid N (N×N)", min_value=2, max_value=11, value=5, step=1,
        key="waf_n"))
    waf_res = wc4.selectbox(
        "Voxel grid density", list(RES_LEVELS.keys()), key="waf_res",
        help="Finer = slower; each grid cell is a full engine run.")
    _wsmc, _wsmv = RES_LEVELS[waf_res]
    # Grid spans the whole wafer; cells outside the disk or below the primary
    # flat are skipped.  Reused by the run loop and the maps below.
    wcoords = np.linspace(-R_mm, R_mm, waf_n)
    Xg, Yg = np.meshgrid(wcoords, wcoords)
    wmask = (Xg ** 2 + Yg ** 2 <= R_mm ** 2) & (Yg >= -d_flat)
    n_on = int(wmask.sum())
    st.caption(f"{waf_size} wafer (R = {R_mm:.1f} mm) • {n_on}/"
               f"{waf_n * waf_n} on-wafer cells at '{waf_res}' density.")

    # Per-evaporation schematic of the fixed source + tilted wafer (always on).
    with st.spinner("Drawing source / wafer-tilt schematic…"):
        figgeo = vv.render_wafer_geometry(params, waf_L)
        st.pyplot(figgeo, use_container_width=True)
        plt.close(figgeo)

    # Cache helpers — mirror _scan_sig / _scan_area; defined locally so the tab
    # is self-contained (no reliance on tab-order name leakage).
    def _wafer_sig(p):
        return (p.mode, p.stack, p.angle1, p.phi1, p.angle2, p.phi2,
                p.tri_angle2, p.tri_phi2, p.tri_angle4, p.tri_phi4,
                p.bridge_len, p.bridge_w, p.bridge_pmma_gap, p.undercut,
                p.t_mma, p.t_pmma, p.manhattan_wx, p.manhattan_wy,
                p.manhattan_theta, p.manhattan_delta, p.manhattan_h,
                _wsmc, _wsmv)

    @st.cache_data(show_spinner=False)
    def _wafer_area(sig, _p):
        r = simulate(_p, max_cells=_wsmc, min_vox=_wsmv)
        _, area, _, _, _ = junction_footprint(r)
        return float(area)

    if st.button("▶ Run wafer map", key="run_wafer",
                 use_container_width=True):
        if n_on == 0:
            st.warning("No grid cell lands on the wafer at this N — "
                       "increase **Grid N (N×N)**.")
        else:
            # Per-evaporation effective (θ′, φ′) over the whole grid — pure math,
            # vectorised, computed once (the engine still runs per on-wafer cell).
            theta_grids, phi_grids = {}, {}
            for lbl, _ta, _pa, th, ph in evap_beams(params):
                tg, pg = wafer_local_angles(th, ph, Xg, Yg, waf_L)
                theta_grids[lbl], phi_grids[lbl] = tg, pg

            warea = np.full((waf_n, waf_n), np.nan)       # [iy, ix]; off-wafer NaN
            done = 0
            prog = st.progress(0.0, text="Simulating wafer positions…")
            for iy in range(waf_n):
                for ix in range(waf_n):
                    if not wmask[iy, ix]:
                        continue                          # skip off-wafer cells
                    q = wafer_params(params, float(Xg[iy, ix]),
                                     float(Yg[iy, ix]), waf_L)
                    warea[iy, ix] = _wafer_area(_wafer_sig(q), q)
                    done += 1
                    prog.progress(done / n_on,
                                  text=f"Simulating… {done}/{n_on}")
            prog.empty()
            st.session_state["_wafermap"] = dict(
                coords=wcoords, areas=warea, theta=theta_grids, phi=phi_grids,
                L=waf_L, R=R_mm, c=c_flat, d=d_flat, size=waf_size, n=waf_n,
                res=waf_res, mode=params.mode, stack=params.stack)

    wm = st.session_state.get("_wafermap")
    if wm and "theta" in wm:                              # new-format result
        warea = wm["areas"]

        # (a) Drawn wafer (matplotlib): area + Ic clipped to the disk + flat.
        figw = vv.render_wafer_map_2d(
            wm["coords"], warea, warea * 1e-4, wm["R"], wm["c"], wm["d"],
            title=(f"{wm['size']} wafer — {wm['stack']}/{wm['mode']}  "
                   f"(L={wm['L']:.0f} mm, {wm['n']}×{wm['n']}, {wm['res']})"))
        st.pyplot(figw, use_container_width=True); plt.close(figw)

        # On-wafer cell mask (finite area), shared by the chart + table.
        Xg, Yg = np.meshgrid(wm["coords"], wm["coords"])
        on = ((Xg ** 2 + Yg ** 2 <= wm["R"] ** 2) & (Yg >= -wm["d"])
              & np.isfinite(warea))

        # (b) Interactive hover heatmap (Altair).  ASCII column keys avoid
        # Vega-Lite field-name escaping; pretty labels live in tooltip titles.
        h = (wm["coords"][1] - wm["coords"][0]) if wm["n"] > 1 else wm["R"]
        cdata = {"x": Xg[on], "y": Yg[on], "area": warea[on],
                 "Ic": warea[on] * 1e-4}
        tips = [alt.Tooltip("x:Q", title="x [mm]", format=".1f"),
                alt.Tooltip("y:Q", title="y [mm]", format=".1f"),
                alt.Tooltip("area:Q", title="area [nm²]", format=".0f"),
                alt.Tooltip("Ic:Q", title="Ic [µA]", format=".4f")]
        for lbl in wm["theta"]:
            key = lbl.replace(" ", "_")
            cdata[f"{key}_th"] = wm["theta"][lbl][on]
            cdata[f"{key}_ph"] = wm["phi"][lbl][on]
            tips.append(alt.Tooltip(f"{key}_th:Q", title=f"{lbl} θ′ [°]",
                                    format=".2f"))
            tips.append(alt.Tooltip(f"{key}_ph:Q", title=f"{lbl} φ′ [°]",
                                    format=".2f"))
        cdf = pd.DataFrame(cdata)
        cdf = cdf.assign(x0=cdf["x"] - h / 2, x1=cdf["x"] + h / 2,
                         y0=cdf["y"] - h / 2, y1=cdf["y"] + h / 2)
        heat = alt.Chart(cdf).mark_rect().encode(
            x=alt.X("x0:Q", title="wafer x  [mm]"), x2="x1:Q",
            y=alt.Y("y0:Q", title="wafer y  [mm]"), y2="y1:Q",
            color=alt.Color("area:Q", title="area [nm²]",
                            scale=alt.Scale(scheme="viridis")),
            tooltip=tips)
        tt = np.linspace(np.arctan2(-wm["d"], wm["c"] / 2),
                         np.arctan2(-wm["d"], -wm["c"] / 2) + 2 * np.pi, 200)
        out = pd.DataFrame({
            "x": np.r_[wm["R"] * np.cos(tt), -wm["c"] / 2, wm["c"] / 2],
            "y": np.r_[wm["R"] * np.sin(tt), -wm["d"], -wm["d"]],
            "o": np.arange(202)})
        outline = alt.Chart(out).mark_line(color="#90A4AE").encode(
            x="x:Q", y="y:Q", order="o:O")
        st.altair_chart((heat + outline).properties(width=480, height=480),
                        use_container_width=False)
        st.caption("Hover any cell for its area, Ic, and per-evaporation "
                   "effective (θ′, φ′).")

        # (c) Sortable table of every on-wafer cell (pretty column names).
        rows = {"x [mm]": Xg[on].round(2), "y [mm]": Yg[on].round(2),
                "area [nm²]": warea[on].round(0),
                "Ic [µA]": (warea[on] * 1e-4).round(4)}
        for lbl in wm["theta"]:
            rows[f"{lbl} θ′ [°]"] = wm["theta"][lbl][on].round(2)
            rows[f"{lbl} φ′ [°]"] = wm["phi"][lbl][on].round(2)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "💾 Download wafer-map table (CSV)", data=df.to_csv(index=False),
            file_name="wafer_map.csv", mime="text/csv")

        # (d) Spread statistics (NaN-safe; over on-wafer cells only).
        amin = float(np.nanmin(warea)); amax = float(np.nanmax(warea))
        amean = float(np.nanmean(warea)); astd = float(np.nanstd(warea))
        rel = (amax - amin) / amean * 100.0 if amean else float("nan")
        cv = astd / amean * 100.0 if amean else float("nan")
        ci = int(np.argmin(np.abs(wm["coords"])))         # centre-most index
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Area min", f"{amin:.0f} nm²")
        s2.metric("Area max", f"{amax:.0f} nm²")
        s3.metric("Area mean", f"{amean:.0f} nm²")
        s4.metric("Area spread", f"{rel:.1f} %",
                  help="(max − min) / mean across the wafer — the JJ-area "
                       "‘ふらつき’.")
        wic = warea * 1e-4
        st.caption(
            f"Centre cell (nominal single-JJ) = {warea[ci, ci]:.0f} nm²  •  "
            f"std/mean = {cv:.1f} %  •  Ic range "
            f"{np.nanmin(wic):.3f}–{np.nanmax(wic):.3f} µA.")
    else:
        st.info("Set the source/grid configuration above and press "
                "**Run wafer map**.")

# ═══ TAB 5: Junction Area ════════════════════════════════════════
with tab5:
    st.subheader("Junction Area & Full Parameter Summary")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Overlap x (engine)",  f"{eng_ox:.0f} nm")
    c2.metric("Overlap y (engine)",  f"{eng_oy:.0f} nm")
    c3.metric("Area A (engine)",     f"{eng_area:.0f} nm²")
    c4.metric("Est. Ic (engine)",    f"{eng_ic:.3f} µA",
              help="Al, 4K: jc=10 kA/cm² (Ambegaokar-Baratoff)")
    c5.metric("Junctions",           f"{eng_njunc}",
              help="Number of spatially separate Al1∩Al2 overlaps")
    j1, j2, j3 = st.columns(3)
    j1.metric("Josephson inductance L_J", _fmt_lj(eng_jj["Lj_nH"]),
              help="L_J = ħ / (2e·Iᶜ)")
    j2.metric("Josephson energy E_J/h", f"{eng_jj['Ej_h_GHz']:.2f} GHz",
              help="E_J = (Φ₀/2π)·Iᶜ = ħ·Iᶜ/2e")
    j3.metric("E_J / k_B", f"{eng_jj['Ej_kB_K']:.2f} K")
    if params.stack == "Trilayer":
        _combo_metrics(eng_combos)
    st.caption(
        f"Engine voxel size = {eng.vox:.1f} nm.  Junction area is the true "
        "Al1∩Al2 overlap (Σ overlapping voxels × voxel²), exact for "
        "non-rectangular junctions."
    )
    st.divider()
    if mode == "Dolan bridge":
        detail = {
            "Mode":                  params.mode,
            "PMMA [nm]":             params.t_pmma,
            "MMA [nm] (vert. gap)":  params.t_mma,
            "Bridge↔PMMA gap [nm]":  params.bridge_pmma_gap,
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
            "Junctions (engine)":    eng_njunc,
            "Estimated Ic (engine) [µA]":  eng_ic,
            "Josephson inductance L_J [nH]": eng_jj["Lj_nH"],
            "Josephson energy E_J/h [GHz]":  eng_jj["Ej_h_GHz"],
            "Josephson energy E_J/k_B [K]":  eng_jj["Ej_kB_K"],
            "Engine voxel [nm]":     eng.vox,
        }
    else:
        detail = {
            "Mode":                  params.mode,
            "θ₁ [°]":                params.angle1,
            "φ₁ [°]":                params.phi1,
            "Metal d₁ [nm]":         params.t_metal1,
            "θ₂ [°]":                params.angle2,
            "φ₂ [°]":                params.phi2,
            "Metal d₂ [nm]":         params.t_metal2,
            "x-arm opening wx [nm]": params.manhattan_wx,
            "y-arm opening wy [nm]": params.manhattan_wy,
            "Overlap x (engine) [nm]":   eng_ox,
            "Overlap y (engine) [nm]":   eng_oy,
            "Junction area (engine) [nm²]": eng_area,
            "Junctions (engine)":    eng_njunc,
            "Estimated Ic (engine) [µA]":  eng_ic,
            "Josephson inductance L_J [nH]": eng_jj["Lj_nH"],
            "Josephson energy E_J/h [GHz]":  eng_jj["Ej_h_GHz"],
            "Josephson energy E_J/k_B [K]":  eng_jj["Ej_kB_K"],
            "Engine voxel [nm]":     eng.vox,
        }
    if params.stack == "Trilayer":
        detail["Stack"] = "Trilayer (Nb/Al/Al/Nb)"
        detail["Nb d₁ [nm] (evap1)"] = params.tri_t1
        detail["Al d₂ [nm] (evap2)"] = params.tri_t2
        detail["Al d₃ [nm] (evap3)"] = params.tri_t3
        detail["Nb d₄ [nm] (evap4)"] = params.tri_t4
        detail["Evap2 θ₂ [°]"] = params.tri_angle2
        detail["Evap2 φ₂ [°]"] = params.tri_phi2
        detail["Evap4 θ₄ [°]"] = params.tri_angle4
        detail["Evap4 φ₄ [°]"] = params.tri_phi4
        for name in _COMBO_ORDER:
            detail[f"Barrier {name} area [nm²]"] = float(
                eng_combos.get(name, {}).get("area", 0.0))
    num_detail = {k:v for k,v in detail.items() if isinstance(v, (int,float))}
    str_detail = {k:v for k,v in detail.items() if isinstance(v, str)}
    rows = [(k, v) for k,v in str_detail.items()]
    rows += [(k, f"{v:.3f}") for k,v in num_detail.items()]
    st.dataframe(
        {"Parameter": [r[0] for r in rows],
         "Value":     [r[1] for r in rows]},
        use_container_width=True, hide_index=True)
    st.download_button(
        "💾 Export parameters + results (re-loadable)",
        data=_build_export(params, eng, eng_area, eng_ox, eng_oy,
                           eng_njunc, eng_ic, eng_juncs, res_level,
                           combos=eng_combos),
        file_name="shadowcast_params.json", mime="application/json",
        help="Re-load this file with the sidebar uploader to restore every "
             "parameter.")
