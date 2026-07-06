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
import json, copy, os, sys, time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

from gds_parser import load_gds_polygons, list_layers
from process_engine import (ProcessParams, shadow_vector, wafer_params,
                            wafer_params_gaussian, wafer_params_source,
                            sample_beam_cloud,
                            evap_beams, wafer_local_angles, wafer_source_dist)
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
    "Ultra-max (heavy)":   (520, 1.5),
    "Extreme (very heavy)": (640, 1.0),
}

# ─── Wafer sizes (SEMI primary-flat specs):  label → (radius_mm, flat_chord_mm) ──
# The primary flat (オリフラ) chord sits at y = −d, d = sqrt(R² − (chord/2)²).
WAFER_SPECS = {
    "2 inch": (25.4, 15.88),
    "3 inch": (38.1, 22.22),
    "4 inch": (50.0, 32.50),
    "6 inch": (75.0, 57.50),
}

# ─── E-beam source raster patterns (finite-source Monte-Carlo) ───────────────
# label → sampler key consumed by process_engine.sample_beam_cloud.  "Rotating line"
# (line spot swept + rotated 10 rpm → centre-peaked disk, areal density ∝ 1/ρ) is the
# default; it is the physically correct profile for the standard recipe.
BEAM_PATTERNS = {
    "Rotating line (disk, 1/ρ)": "rotline",
    "Uniform disk":              "uniform",
    "Gaussian":                  "gaussian",
    "Point":                     "point",
}


def _beam_pattern_controls(prefix, container=st):
    """Render the beam-pattern picker + its size input, returning ``(pattern_key,
    size)`` for ``sample_beam_cloud`` / ``wafer_params_source``.  ``size`` is σ for
    Gaussian, else the disk diameter; all patterns are isotropic (no orientation).
    Widget keys are ``prefix``-scoped so the single-JJ and wafer-map pickers stay
    independent."""
    lbl = container.selectbox("Beam pattern", list(BEAM_PATTERNS),
                              key=f"{prefix}_pattern",
                              help="Source intensity shape. 'Rotating line' (line spot "
                                   "swept + rotated 10 rpm) is the standard recipe — a "
                                   "centre-peaked disk, areal density ∝ 1/ρ.")
    pat = BEAM_PATTERNS[lbl]
    size = 0.0
    if pat == "gaussian":
        size = float(container.number_input("σ [mm]", 0.0, 50.0, 2.0, 0.5,
                                            key=f"{prefix}_sigma",
                                            help="Source r.m.s. transverse size."))
    elif pat in ("uniform", "rotline"):
        size = float(container.number_input(
            "Source size [mm] (diameter)", 0.0, 60.0, 12.0, 0.5,
            key=f"{prefix}_size",
            help="Full disk diameter (≈10–15 mm typical)."))
    return pat, size


def _source_dist_charts(pattern, size, rng_seed=1, n=4000):
    """Altair (scatter, radial-profile) pair visualising the source-plane intensity
    distribution for the chosen beam ``pattern`` — the 2-D sample cloud and its
    probability vs radius ρ (flat for rotating-line, rising ∝ρ for uniform disk,
    Rayleigh-peaked for Gaussian)."""
    cloud = sample_beam_cloud(pattern, size, n, np.random.default_rng(rng_seed))
    lim = max(1.0, float(np.abs(cloud).max()) * 1.1)
    df = pd.DataFrame({"x": cloud[:, 0], "y": cloud[:, 1],
                       "rho": np.hypot(cloud[:, 0], cloud[:, 1])})
    scatter = alt.Chart(df).mark_circle(size=8, opacity=0.25, color="#64B5F6").encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-lim, lim]), title="source x [mm]"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-lim, lim]), title="source y [mm]")
        ).properties(width=320, height=320,
                     title=f"Source-plane intensity — {pattern}")
    radial = alt.Chart(df).mark_bar(color="#80CBC4").encode(
        x=alt.X("rho:Q", bin=alt.Bin(maxbins=30), title="radius ρ [mm]"),
        y=alt.Y("count()", title="samples ∝ probability")
        ).properties(width=320, height=320, title="Radial profile P(ρ)")
    return scatter, radial


def _beam_angle_meta(p):
    """``(labels, theta_nom_signed)`` for the active evaporations of ``p``.
    ``labels`` = ['θ1','φ1','θ2','φ2', …] (1-based over ``evap_beams`` order);
    ``theta_nom_signed`` = each evaporation's signed nominal θ (used to map the
    engine's θ≥0 convention back to the user's sidebar convention)."""
    labels, thnom = [], []
    for j, (_lbl, _ta, _pa, thn, _phn) in enumerate(evap_beams(p)):
        labels += [f"θ{j + 1}", f"φ{j + 1}"]
        thnom.append(float(thn))
    return labels, thnom


def _beam_angle_row(q, p):
    """Flat ``[θ1, φ1, θ2, φ2, …]`` of the perturbed angles ``q`` actually sets
    (engine convention, θ≥0)."""
    out = []
    for _lbl, ta, pa, _thn, _phn in evap_beams(p):
        out += [float(getattr(q, ta)), float(getattr(q, pa))]
    return out


def _angle_distribution_ui(key, A, labels, theta_nom, area=None,
                           area_label="JJ area [nm²]"):
    """1-D / 2-D / correlation viewer for an ``(n_samples × 2·nbeams)`` raw angle
    array ``A``.  Re-expresses angles in the user's nominal (signed) convention:
    flip the sign of any θ whose nominal is negative (φ → φ−180), then unwrap each
    φ column about its median so tiny clusters stay continuous (no ±180 split).
    Adds JJ area as a selectable variable when given.  ``key`` namespaces the
    widgets so multiple instances coexist.  Safe internal column names (s0,s1,…)
    sidestep Altair's shorthand parser (θ/φ/brackets)."""
    A = np.asarray(A, float)
    cols, titles = {}, {}
    for j in range(len(theta_nom)):
        th = A[:, 2 * j].copy(); ph = A[:, 2 * j + 1].copy()
        if theta_nom[j] < 0:                     # back to the user's signed convention
            th = -th; ph = ph - 180.0
        med = float(np.median(ph))               # unwrap φ to stay continuous
        ph = ((ph - med + 180.0) % 360.0) - 180.0 + med
        cols[f"s{2 * j}"] = th;     titles[f"s{2 * j}"] = f"{labels[2 * j]} [°]"
        cols[f"s{2 * j + 1}"] = ph; titles[f"s{2 * j + 1}"] = f"{labels[2 * j + 1]} [°]"
    if area is not None:
        cols["area"] = np.asarray(area, float); titles["area"] = area_label
    df = pd.DataFrame(cols)
    inv = {v: k for k, v in titles.items()}      # display title → safe column name
    names = list(titles.values())
    view = st.radio("View", ["1-D histogram", "2-D scatter", "Correlation matrix"],
                    horizontal=True, key=f"{key}_view")
    if view == "1-D histogram":
        v = st.selectbox("Variable", names, key=f"{key}_1d")
        ch = alt.Chart(df).mark_bar(color="#64B5F6").encode(
            x=alt.X(field=inv[v], type="quantitative",
                    bin=alt.Bin(maxbins=30), title=v),
            y=alt.Y("count()", title="samples"))
        st.altair_chart(ch.properties(height=320), use_container_width=True)
    elif view == "2-D scatter":
        c1, c2 = st.columns(2)
        vx = c1.selectbox("X", names, index=0, key=f"{key}_x")
        vy = c2.selectbox("Y", names, index=min(1, len(names) - 1), key=f"{key}_y")
        sd = float(df[inv[vx]].std() * df[inv[vy]].std())
        r = (float(np.corrcoef(df[inv[vx]], df[inv[vy]])[0, 1])
             if sd > 0 else float("nan"))
        ch = alt.Chart(df).mark_circle(size=22, opacity=0.5, color="#CE93D8").encode(
            x=alt.X(field=inv[vx], type="quantitative",
                    scale=alt.Scale(zero=False), title=vx),
            y=alt.Y(field=inv[vy], type="quantitative",
                    scale=alt.Scale(zero=False), title=vy))
        st.altair_chart(ch.properties(width=420, height=420),
                        use_container_width=False)
        st.caption(f"Pearson r = {r:.3f}  (n = {len(df)} MC samples).")
    else:
        with np.errstate(invalid="ignore"):
            M = np.corrcoef(np.column_stack([df[inv[v]] for v in names]).T)
        M = np.atleast_2d(np.nan_to_num(M, nan=0.0))
        long = pd.DataFrame([{"a": names[i], "b": names[j], "r": float(M[i, j])}
                             for i in range(len(names)) for j in range(len(names))])
        heat = alt.Chart(long).mark_rect().encode(
            x=alt.X("a:N", title=None, sort=names),
            y=alt.Y("b:N", title=None, sort=names),
            color=alt.Color("r:Q",
                            scale=alt.Scale(scheme="redblue", domain=[-1, 1])))
        txt = alt.Chart(long).mark_text(baseline="middle").encode(
            x=alt.X("a:N", sort=names), y=alt.Y("b:N", sort=names),
            text=alt.Text("r:Q", format=".2f"), color=alt.value("black"))
        st.altair_chart((heat + txt).properties(width=420, height=420),
                        use_container_width=False)
        st.caption("Pearson correlation. With an isotropic source each evaporation's "
                   "angles are drawn independently (angle–angle ≈ 0); correlations "
                   "between an angle and JJ area show which beams drive the area.")

# Sidebar widget keys.  Mode-specific widgets get DISTINCT keys (prefixed) so
# switching modes can never feed an out-of-range value into a shared slider.
_SHARED_KEYS = ["t_pmma", "t_mma", "undercut", "resist_round",
                "angle1", "phi1", "t_metal1"]
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
    "t_pmma": (100, 2000), "t_mma": (100, 1500), "undercut": (0, 500),
    "resist_round": (0, 200), "jc_al": (1, 1e6),
    "soft_L": (20.0, 2000.0),
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
    if pdict.get("resist_round_method") in ("analytic", "voxel"):
        st.session_state["resist_round_method"] = pdict["resist_round_method"]; applied += 1
    if pdict.get("sidewall") is not None:
        st.session_state["sidewall"] = bool(pdict["sidewall"]); applied += 1
    if pdict.get("jc_al") is not None:
        v = float(pdict["jc_al"])
        if 1.0 <= v <= 1e6:
            st.session_state["jc_al"] = v; applied += 1
    if pdict.get("jj_walls") is not None:
        st.session_state["jj_walls"] = bool(pdict["jj_walls"]); applied += 1
    if pdict.get("soft_edge") is not None:
        st.session_state["soft_edge"] = bool(pdict["soft_edge"]); applied += 1
    if pdict.get("soft_pattern") is not None:        # stored as the key (e.g. 'rotline')
        _key2lbl = {v: k for k, v in BEAM_PATTERNS.items()}
        _lbl = _key2lbl.get(pdict["soft_pattern"])
        if _lbl is not None:
            st.session_state["soft_pattern"] = _lbl; applied += 1
    if pdict.get("soft_size") is not None:           # fills both size widgets
        v = float(pdict["soft_size"])
        st.session_state["soft_size"] = v
        st.session_state["soft_sigma"] = v; applied += 1
    if pdict.get("soft_L") is not None:
        v = float(pdict["soft_L"])
        if 20.0 <= v <= 2000.0:
            st.session_state["soft_L"] = v; applied += 1
    if pdict.get("soft_rays") is not None:
        v = int(pdict["soft_rays"])
        if 4 <= v <= 200:
            st.session_state["soft_rays"] = v; applied += 1
    if pdict.get("soft_supersample_xy") is not None:
        v = int(pdict["soft_supersample_xy"])
        if 1 <= v <= 10:
            st.session_state["soft_supersample_xy"] = v; applied += 1
    if pdict.get("soft_supersample_z") is not None:
        v = int(pdict["soft_supersample_z"])
        if 1 <= v <= 10:
            st.session_state["soft_supersample_z"] = v; applied += 1
    if raydict and raydict.get("resolution") in RES_LEVELS:
        st.session_state["res_level"] = raydict["resolution"]; applied += 1
    return applied


# ─── Josephson-junction electrical quantities ───────────────────────
_E_CHG = 1.602176634e-19         # elementary charge      [C]
_H_PL  = 6.62607015e-34          # Planck constant         [J·s]
_HBAR  = _H_PL / (2 * np.pi)     # reduced Planck constant [J·s]
_PHI0  = _H_PL / (2 * _E_CHG)    # magnetic flux quantum   [Wb]
_KB    = 1.380649e-23            # Boltzmann constant      [J/K]
_DELTA_AL = 1.764 * _KB * 1.2    # Al superconducting gap  [J] (Tc ≈ 1.2 K) ≈ 0.18 meV


def _rn_from_ic_uA(ic_uA):
    """Normal-state resistance R_n [Ω] from Ic via Ambegaokar–Baratoff (T→0):
    ``Ic·R_n = πΔ/2e`` ⇒ ``R_n = πΔ/(2e·Ic)``.  Scalar or array; Ic ≤ 0 → ∞."""
    ic = np.asarray(ic_uA, float) * 1e-6           # A
    with np.errstate(divide="ignore", invalid="ignore"):
        rn = np.pi * _DELTA_AL / (2.0 * _E_CHG * ic)
    return np.where(ic > 0, rn, np.inf)


def jj_electrical(ic_uA):
    """Josephson inductance, energy & normal-state resistance derived from Ic.

    L_J = ħ / (2e·Ic),   E_J = (Φ₀/2π)·Ic = ħ·Ic/2e,
    R_n = πΔ/(2e·Ic)  (Ambegaokar–Baratoff, Al gap Δ at T→0).
    Returns L_J [nH], E_J [J], E_J/h [GHz], E_J/kB [K] and R_n [Ω].  For Ic ≤ 0
    (open circuit) L_J and R_n are infinite and E_J is zero."""
    ic = float(ic_uA) * 1e-6                       # A
    if ic <= 0:
        return dict(Lj_nH=float("inf"), Ej_J=0.0, Ej_h_GHz=0.0, Ej_kB_K=0.0,
                    Rn_ohm=float("inf"))
    Lj = _HBAR / (2 * _E_CHG * ic)                 # H
    Ej = (_PHI0 / (2 * np.pi)) * ic                # J
    return dict(Lj_nH=Lj * 1e9, Ej_J=Ej,
                Ej_h_GHz=Ej / _H_PL / 1e9, Ej_kB_K=Ej / _KB,
                Rn_ohm=float(_rn_from_ic_uA(ic_uA)))


def _fmt_lj(lj_nH):
    """Human-readable Josephson inductance (nH / µH, or ∞ for an open junction)."""
    if not np.isfinite(lj_nH):
        return "∞ (open)"
    if lj_nH >= 1000.0:
        return f"{lj_nH / 1000.0:.3f} µH"
    return f"{lj_nH:.3f} nH"


def _fmt_rn(ohm):
    """Human-readable normal-state resistance (Ω / kΩ, or ∞ for an open junction)."""
    if not np.isfinite(ohm):
        return "∞ (open)"
    if ohm >= 1000.0:
        return f"{ohm / 1000.0:.2f} kΩ"
    return f"{ohm:.1f} Ω"


def _fmt_sig(v, sig=2):
    """Format to `sig` significant figures, plain (no sci-notation) for sensible
    magnitudes; blank for non-finite (open/off-wafer)."""
    if v is None or not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    import math
    d = (sig - 1) - math.floor(math.log10(abs(v)))   # decimals for `sig` sig figs
    return f"{round(v, d):g}"


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
    "t_pmma": 250, "t_mma": 900, "undercut": 150, "resist_round": 0,
    "resist_round_method": "analytic",
    "angle1": -24, "phi1": 0, "t_metal1": 30,
    "d_angle2": 24, "d_phi2": 0, "d_tmetal2": 30,
    "d_bridge_len": 250, "d_bridge_w": 250, "d_bridge_pmma_gap": 0,
    "m_angle2": 60, "m_phi2": 90, "m_tmetal2": 30,
    "m_wx": 600, "m_wy": 600,
    "tri_t1": 80, "tri_t2": 10, "tri_t3": 10, "tri_t4": 150,
    "tri_a2": -24, "tri_p2": 0, "tri_a4": 24, "tri_p4": 0,
    "tri_link2": True, "tri_link4": True,
    "sidewall": False,
    "jc_al": 1000,
    "jj_walls": False,
    "soft_edge": False,
    "soft_pattern": "Rotating line (disk, 1/ρ)",   # _beam_pattern_controls selectbox label
    "soft_size": 12.0, "soft_sigma": 2.0, "soft_L": 550.0, "soft_rays": 24,
    "soft_supersample_xy": 1, "soft_supersample_z": 1,
    "res_level": "Standard (fast)",
}


def _reset_defaults():
    """Restore every process-parameter widget to its default value."""
    for k, v in _PARAM_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("_imported_sig", None)   # allow re-loading later


def _on_stack_change():
    """On a real switch to Trilayer: re-link evap-2/4 to their electrode primary and
    carry the current evap-1 / evap-3 tilts into the evap-2/4 controls, so the angle
    info is preserved in every case (links on, or later unlinked — including a
    re-switch after the primary changed).  Fires only on real user interaction, so the
    file-import path that intentionally unlinks them (``_apply_loaded_params``) is
    preserved."""
    if st.session_state.get("stack") != "Trilayer":
        return
    st.session_state["tri_link2"] = True
    st.session_state["tri_link4"] = True
    # Drop any stale independent evap-2/4 angles so the next time they are unlinked
    # they re-seed from the *current* evap-1 / evap-3 tilts (``_tri_linked_tilt``'s
    # ``setdefault``).  Popping (vs assigning) avoids Streamlit reverting an
    # unrendered widget key to its old backed-up value while the link stays on.
    for k in ("tri_a2", "tri_p2", "tri_a4", "tri_p4"):
        st.session_state.pop(k, None)


def _build_export(params, eng, area, ox, oy, njunc, ic, juncs, res_level,
                  combos=None, jc_al=1000.0, jj_walls=False):
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
    params_dict = asdict(params)
    params_dict["jc_al"] = float(jc_al)
    params_dict["jj_walls"] = bool(jj_walls)
    out = {
        "shadowcast": "v6",
        "mode": params.mode,
        "stack": params.stack,
        "parameters": params_dict,
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
    # ── Run gate ──────────────────────────────────────────────────
    # The single-JJ engine runs only when this is pressed (or on first load),
    # so dragging a slider does NOT re-simulate.  Display / measurement controls
    # (Jc, sidewall-area toggle, cross-section view, wafer colour) stay live.
    run_sim = st.button("▶ Run simulation", type="primary",
                        use_container_width=True,
                        help="Run the 3-D engine with the current sidebar "
                             "settings.  Geometry / angle / resolution changes "
                             "take effect only when you press this.")
    _stale_box = st.empty()    # filled below with a 'parameters changed' notice

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
        key="stack", on_change=_on_stack_change,
        help="Bilayer: evap 1 → oxidation → evap 2.  "
             "Trilayer: Nb→Al → oxidation → Al→Nb (Nb/Al/Al/Nb), forming a "
             "junction whose barrier metals are reported as Nb-Al / Al-Al / Nb-Nb.")

    # Trilayer sublayer values.  The engine reads these only when
    # stack == "Trilayer"; in that case the per-evaporation blocks below set
    # them.  Bilayer leaves them at these inert defaults.
    tri_t1, tri_t2, tri_t3, tri_t4 = 80.0, 10.0, 10.0, 150.0
    tri_angle2 = tri_phi2 = tri_angle4 = tri_phi4 = 0.0

    st.subheader("Bilayer resist")
    if mode == "Dolan bridge":
        st.caption("Recipe arxiv:2101.01453 — PMMA A-4 ≈250 nm / MMA EL-13 ≈900 nm")
    else:
        st.caption("Manhattan: MMA = lower undercut sublayer · "
                   "PMMA = upper imaging resist (recipe: ~1800 nm)")
    st.session_state.setdefault("t_pmma", 250)
    t_pmma = st.slider("PMMA [nm]  (top, no undercut)", 100, 2000,
                       step=25, key="t_pmma")
    st.session_state.setdefault("t_mma", 900)
    t_mma = st.slider("MMA [nm]  (bottom = bridge height / vertical gap)",
                      100, 1500, step=25, key="t_mma",
                      help=("MMA bottom-layer thickness.  The bridge underside "
                            "sits at z = MMA height, so this sets the vertical "
                            "shadow gap.  Junction overlap ≈ 2·MMA·tanθ − bridge width."
                            if mode == "Dolan bridge" else
                            "Lower undercut sublayer thickness.  The undercut shelf "
                            "lets the metal lift off cleanly.  Manhattan: typically "
                            "400–600 nm."))
    st.session_state.setdefault("undercut", 150)
    undercut = st.slider("MMA undercut u [nm]  (one-sided)", 0, 500,
                         step=10, key="undercut")
    st.session_state.setdefault("resist_round", 0)
    resist_round = st.slider(
        "Resist corner rounding r [nm]", 0, 200, step=5, key="resist_round",
        help="Round the resist opening's top lip and bottom foot with a fillet "
             "of this radius (0 = sharp corners).  The wall flares near the top "
             "and floor, so it shifts the shadow and the junction area.")
    resist_round_method = "analytic"
    if resist_round > 0:
        st.session_state.setdefault("resist_round_method", "analytic")
        resist_round_method = st.radio(
            "Rounding method", ["analytic", "voxel"],
            format_func=lambda v: "Analytic (default, fast)" if v == "analytic"
                                  else "Voxel stack (legacy, slow)",
            key="resist_round_method", horizontal=True,
            help="Analytic: exact continuous quarter-circle, solved directly "
                 "per ray (fast, no K-slab approximation).  Voxel stack: the "
                 "original method — approximates the fillet with ~10-20 thin "
                 "box layers, which is **significantly slower** (more boxes "
                 "to test against every ray) and only an approximation of "
                 "the analytic curve.  Kept as a fallback / cross-check.")
    if mode == "Dolan bridge":
        st.caption(f"Total resist: {t_pmma+t_mma} nm  ·  vertical shadow gap = MMA = {t_mma} nm")
    else:
        st.caption(f"Lower sublayer (MMA): {t_mma} nm  ·  "
                   f"Upper imaging (PMMA): {t_pmma} nm  ·  "
                   f"Total: {t_pmma+t_mma} nm")

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
        manhattan_h = float(t_pmma)   # upper imaging resist only; lower sublayer = t_mma

    st.divider()
    st.subheader("Ray-scan resolution")
    st.session_state.setdefault("res_level", "Standard (fast)")
    res_level = st.selectbox(
        "Voxel grid density", list(RES_LEVELS.keys()), key="res_level",
        help="Finer = the tilted beam is ray-traced into a denser voxel grid "
             "(smaller voxels) for sharper metal / junction edges, at the cost "
             "of speed and memory.")
    _max_cells, _min_vox = RES_LEVELS[res_level]
    if _max_cells >= 520:
        st.caption("⚠ Very fine grid — high memory & run time (cells grow ≈ as "
                   "max_cells³).")

    st.divider()
    st.subheader("⚙ Process options")
    sidewall = st.checkbox(
        "Side-wall effect (1st-evap coating narrows later evaporations)",
        key="sidewall",
        help="The first evaporation also coats the resist sidewall, narrowing the "
             "opening seen by later evaporations (≈ the deposited thickness, "
             "auto-varying with the local incident angle across the wafer — "
             "Jpn. J. Appl. Phys. aca256).  Opt-in; off = ideal openings.")
    st.session_state.setdefault("jc_al", 1000)
    jc_al = float(st.number_input(
        "Critical current density Jc [A/cm²]",
        min_value=1.0, max_value=1e6, step=100.0, key="jc_al",
        help="Al-AlOx-Al junction critical current density.  "
             "Typical range: 100–10 000 A/cm²  (qubit-grade: ~1 kA/cm²).  "
             "Ic = Jc × area;  R_N = πΔ/(2e·Ic) (Ambegaokar–Baratoff, Δ ≈ 0.18 meV)."))
    ic_factor = jc_al * 1e-8  # nm² → µA  (Jc[A/cm²] × 1e-14 cm²/nm² × 1e6 µA/A)
    st.session_state.setdefault("jj_walls", False)
    jj_walls = st.checkbox(
        "Count sidewall (vertical) junction area", key="jj_walls",
        help="Off (default): junction area = the horizontal floor overlap only "
             "(electrode 2 on top of electrode 1 across the oxide), matching the "
             "expected planar junction.  On: also count the vertical M-O-M "
             "barrier where metal climbs the resist sidewall (full 3-D area, "
             "larger).  When on, the floor-only and sidewall-only areas are "
             "shown separately.")
    st.session_state.setdefault("soft_edge", False)
    soft_edge = st.checkbox(
        "Soft edge (finite-source penumbra)", key="soft_edge",
        help="Model the real Plassys source: occlusion is integrated over the "
             "e-beam raster pattern at the throw distance, tapering the film "
             "thickness near the shadow edge (penumbra ≈ source size / L).  "
             "Combined with a rounded resist lip this gives a rounded metal edge.  "
             "Visible at finer resolution (film several voxels thick); slower.")
    soft_pat, soft_size, soft_L, soft_rays, soft_supersample_xy, soft_supersample_z = \
        "rotline", 12.0, 550.0, 24, 1, 1
    if soft_edge:
        soft_pat, soft_size = _beam_pattern_controls("soft")
        st.session_state.setdefault("soft_L", 550.0)
        soft_L = float(st.number_input(
            "Throw distance L [mm]", 20.0, 2000.0, step=10.0, key="soft_L",
            help="Source→sample distance (Plassys ≈ 550 mm).  Penumbra ≈ size/L."))
        st.session_state.setdefault("soft_rays", 24)
        soft_rays = int(st.number_input(
            "Source-cloud rays K", 4, 200, step=4, key="soft_rays",
            help="Number of sampled directions across the source used to "
                 "integrate the penumbra coverage at each edge voxel.  Higher = "
                 "smoother taper / finer coverage gradation (coverage is only "
                 "resolvable in steps of 1/K), at a roughly linear cost in "
                 "soft-edge runtime."))
        st.session_state.setdefault("soft_supersample_xy", 1)
        soft_supersample_xy = int(st.number_input(
            "Lateral sub-sampling n (x-y, n×n per cell)", 1, 10, step=1, key="soft_supersample_xy",
            help="Sample an n×n sub-grid of lateral (xy) positions within each "
                 "band voxel for the resist-occlusion test, instead of just the "
                 "voxel centre — smooths the in-plane footprint boundary at the "
                 "edge.  1 = centre point only (unchanged).  Cost scales as n² "
                 "on top of the ray count above; does not change the voxel grid "
                 "or thickness-step resolution (that's set by grid density)."))
        st.session_state.setdefault("soft_supersample_z", 1)
        soft_supersample_z = int(st.number_input(
            "Vertical sub-sampling n (z, n per cell)", 1, 10, step=1, key="soft_supersample_z",
            help="Sample n positions through each band voxel's z-extent for the "
                 "same resist-occlusion test, instead of just the cell-centre z — "
                 "smooths the through-thickness taper.  1 = centre z only "
                 "(unchanged).  Cost scales linearly with n on top of the lateral "
                 "sub-sampling and ray count above.  Only affects the main "
                 "coverage/thickness calculation, not the fine in-plane diagnostic "
                 "cross-section (which stays lateral-only)."))
        _n_sub = soft_supersample_xy * soft_supersample_xy * soft_supersample_z
        _half = np.degrees(np.arctan((soft_size / 2.0) / max(soft_L, 1e-9)))
        st.caption(f"Source: {soft_pat} • {soft_size:.1f} mm at {soft_L:.0f} mm • "
                   f"{soft_rays} rays × {soft_supersample_xy}² xy × {soft_supersample_z} z "
                   f"sub-samples ⇒ angular half-size ≈ {_half:.2f}°  (coverage "
                   f"resolution ≈ 1/{soft_rays * _n_sub}).")
        if soft_rays > 48 or _n_sub > 8:
            st.caption("⚠ High ray count / sub-sampling — soft-edge cost scales "
                       "roughly linearly with rays × n_xy² × n_z.")

    st.divider()
    st.subheader("Display")
    show_shadow   = st.checkbox("Show shadow deposits (top view)", True)
    show_undercut = st.checkbox("Show undercut regions (top view)", True)

    st.divider()
    st.subheader("🎲 Finite e-beam source — Monte-Carlo")
    jj_src = st.checkbox(
        "Enable beam-pattern JJ-area statistics", key="jj_src",
        help="Model the e-beam source with a finite spatial spread set by its raster "
             "pattern (rotating line / uniform disk / Gaussian / point) instead of an "
             "ideal point. Press Run to execute N_mc engine simulations — each "
             "independently jitters every evaporation's beam angle — and build the "
             "JJ-area distribution (mean ± σ). Results appear in the 🎲 Source MC tab.")
    if jj_src:
        jj_pat, jj_size = _beam_pattern_controls("jj")
        jc1, jc2 = st.columns(2)
        jj_L = float(jc1.number_input("Throw distance L [mm]", 20.0, 2000.0,
                                      550.0, 10.0, key="jj_L",
                                      help="Source→sample distance; spread ≈ size/L."))
        jj_nmc = int(jc2.number_input("Monte-Carlo samples", 2, 1000, 50, 1,
                                      key="jj_nmc", help="Number of engine runs."))
        _spread = (jj_size if jj_pat == "gaussian" else jj_size / 2.0) / jj_L
        st.caption(f"Pattern: {jj_pat} • angular spread ≈ {np.degrees(_spread):.3f}° "
                   f"• {jj_nmc} engine runs at {res_level}.")
        run_jj_mc = st.button("▶ Run source Monte-Carlo", key="run_jj_mc",
                              use_container_width=True)
    else:
        jj_pat, jj_size, jj_L, jj_nmc, run_jj_mc = "point", 0.0, 550.0, 0, False

params = ProcessParams(
    t_pmma=t_pmma, t_mma=t_mma, undercut=undercut, resist_round=resist_round,
    resist_round_method=resist_round_method,
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
    sidewall=sidewall,
    soft_edge=soft_edge, soft_pattern=soft_pat, soft_size=soft_size, soft_L=soft_L,
    soft_rays=soft_rays, soft_supersample_xy=soft_supersample_xy,
    soft_supersample_z=soft_supersample_z,
)
# ─── 3D physical deposition engine (source of truth) ──────────────
def _engine_cached(ekey):
    """Return the cached engine bundle for ``ekey`` if it is the latest one, else
    None.  Manual (session-state) cache — keyed by the engine signature, holds the
    most recent result — so the heavy compute can take a live-progress callback
    (a ``st.cache_data`` function may not call st elements, which the ETA bar does)."""
    c = st.session_state.get("_engine_cache")
    return c["out"] if (c is not None and c.get("ekey") == ekey) else None


def _run_engine(ekey, params, max_cells, min_vox, progress=None):
    """Pre-compute every per-simulation quantity ONCE per ``ekey``: voxel result,
    grounded mask, floor/full junction footprints + combos.  Cached in
    session_state (latest only) so display-only reruns are instant and the
    sidewall-area toggle just selects floor/full.  ``progress`` = ``cb(frac,label)``
    driven from the deposition iterations for the live ETA, then handed to
    ``vv._grounded_metal`` for a second timed phase (the lift-off connectivity
    flood fill, which can take as long as the deposits themselves)."""
    hit = _engine_cached(ekey)
    if hit is not None:
        return hit
    r = simulate(params, max_cells=max_cells, min_vox=min_vox, record=True,
                 progress=progress)
    if progress is not None:
        progress(0.0, "_phase2_start_")         # reset the ETA clock for the flood fill
    vv._grounded_metal(r, progress=progress)    # populate r.meta['_grounded'] once
    full  = junction_footprint(r, include_sidewalls=True)    # (jm,area,ox,oy,juncs)
    floor = junction_footprint(r, include_sidewalls=False)
    combos_full  = junction_combos(r, include_sidewalls=True)
    combos_floor = junction_combos(r, include_sidewalls=False)
    out = (r, full, floor, combos_full, combos_floor)
    st.session_state["_engine_cache"] = {"ekey": ekey, "out": out}
    return out

def _ekey_for(p):
    """Engine cache-key tuple for a ProcessParams at the current sidebar
    resolution (mirrors exactly the fields `simulate` consumes)."""
    return (p.mode, p.t_pmma, p.t_mma, p.undercut, p.angle1, p.phi1, p.t_metal1,
            p.angle2, p.phi2, p.t_metal2, p.bridge_len, p.bridge_w,
            p.bridge_pmma_gap, p.manhattan_wx, p.manhattan_wy, p.manhattan_theta,
            p.manhattan_delta, p.manhattan_h, _max_cells, _min_vox, p.stack,
            p.tri_t1, p.tri_t2, p.tri_t3, p.tri_t4, p.tri_angle2, p.tri_phi2,
            p.tri_angle4, p.tri_phi4, getattr(p, "sidewall", False),
            getattr(p, "resist_round", 0.0),
            getattr(p, "resist_round_method", "analytic"),
            getattr(p, "soft_edge", False), getattr(p, "soft_pattern", "rotline"),
            getattr(p, "soft_size", 12.0), getattr(p, "soft_L", 550.0),
            getattr(p, "soft_rays", 24), getattr(p, "soft_supersample_xy", 1),
            getattr(p, "soft_supersample_z", 1))

@st.cache_data(show_spinner=False)
def _mc_area(sig, _p):
    """Lean engine scorer for the finite-source Monte-Carlo — JJ area only
    (skips junction_combos); cached per `sig` (= _ekey_for(_p)).  Returns
    ``(floor_area, full_area)`` so the sidewall toggle can pick without re-sim."""
    r = simulate(_p, max_cells=_max_cells, min_vox=_min_vox)
    _, a_full, _, _, _ = junction_footprint(r, include_sidewalls=True)
    _, a_floor, _, _, _ = junction_footprint(r, include_sidewalls=False)
    return float(a_floor), float(a_full)

# ── Run gate ──────────────────────────────────────────────────────
# Commit the simulation inputs only when ▶ Run is pressed (or on first load).
# Until then the engine re-uses the last committed params, so dragging a slider
# never re-simulates.  Jc / sidewall-area / view controls are NOT part of the
# commit, so they keep updating live (no re-sim needed).
_live_sig = _ekey_for(params)
_committed = st.session_state.get("_committed")
if run_sim or _committed is None:
    _committed = {"params": params, "max_cells": _max_cells, "min_vox": _min_vox,
                  "sig": _live_sig}
    st.session_state["_committed"] = _committed
params = _committed["params"]
_max_cells = _committed["max_cells"]; _min_vox = _committed["min_vox"]
if _live_sig != _committed["sig"]:
    _stale_box.warning("⚠ Inputs changed — press **▶ Run simulation** to update.")

res = compute_junction_area(params)
ekey = _ekey_for(params)
# Live ETA driven by the deposition iterations: the engine ticks once per
# occlusion pass; elapsed ÷ fraction-done → approximate time-to-finish.  Only show
# the bar when we will actually compute (cache miss); display-only reruns are
# instant cache hits and skip it.
if _engine_cached(ekey) is None:
    _bar = st.progress(0.0, text="Running 3D engine…")
    _t0 = [time.perf_counter()]                 # list: mutable from the closure below
    _last = [0.0]                               # throttle UI updates

    def _run_progress(frac, label=""):
        now = time.perf_counter()
        if label == "_phase2_start_":            # reset the ETA clock for a new phase
            _t0[0] = now; _last[0] = 0.0
            return
        if frac <= 0 or (now - _last[0] < 0.15 and frac < 1.0):
            return
        _last[0] = now
        el = now - _t0[0]
        rem = el * (1.0 - frac) / frac
        _bar.progress(min(max(frac, 0.0), 1.0),
                      text=f"{label}…  ≈ {rem:.0f}s remaining "
                           f"(≈ {el / frac:.0f}s total)")
else:
    _bar = None
    _run_progress = None

eng, _full, _floor, _combos_full, _combos_floor = _run_engine(
    ekey, params, _max_cells, _min_vox, progress=_run_progress)
if _bar is not None:
    _bar.empty()
# Floor / full junction footprints were measured once inside _run_engine; the
# sidewall-area toggle (live) just selects which to show — no re-measure here.
eng_jm_full,  eng_area_full,  _oxf, _oyf, eng_juncs_full = _full
eng_jm_floor, eng_area_floor, _ox0, _oy0, eng_juncs_floor = _floor
eng_area_walls = max(eng_area_full - eng_area_floor, 0.0)
if jj_walls:
    eng_jm, eng_area, eng_ox, eng_oy, eng_juncs = (
        eng_jm_full, eng_area_full, _oxf, _oyf, eng_juncs_full)
    eng_combos, eng_combo_map = _combos_full
else:
    eng_jm, eng_area, eng_ox, eng_oy, eng_juncs = (
        eng_jm_floor, eng_area_floor, _ox0, _oy0, eng_juncs_floor)
    eng_combos, eng_combo_map = _combos_floor
eng_njunc = len(eng_juncs)
# Engine-based critical current via Ambegaokar-Baratoff (Jc is live, no re-sim).
# Ic[µA] = area_nm2 × jc_al[A/cm²] × 1e-8  (= area × 1e-14 cm²/nm² × 1e6 µA/A × jc)
eng_ic = eng_area * ic_factor
# Josephson inductance L_J and energy E_J derived from that critical current.
eng_jj = jj_electrical(eng_ic)

# ── Finite e-beam source Monte-Carlo (opt-in, button-triggered) ──
# Reuses wafer_params_source at wafer-centre (X=Y=0): each draw independently
# jitters every evaporation's beam angle by the chosen beam pattern's spread, then
# scores the JJ area with the same cached engine.  Nominal point-source run above
# is untouched; results land in st.session_state["_jjmc"] for the Source MC tab.
if run_jj_mc:
    rng = np.random.default_rng(0)            # fixed seed → reproducible/cacheable
    smp = np.empty(jj_nmc)
    _alabels, _athnom = _beam_angle_meta(params)
    jang = np.empty((jj_nmc, len(_alabels)))  # perturbed (θ,φ) per evap, per sample
    prog = st.sidebar.progress(0.0, text="Source Monte-Carlo…")
    for m in range(jj_nmc):
        q = wafer_params_source(params, 0.0, 0.0, jj_L, jj_pat, jj_size, rng)
        _af, _afull = _mc_area(_ekey_for(q), q)
        smp[m] = _afull if jj_walls else _af
        jang[m] = _beam_angle_row(q, params)
        prog.progress((m + 1) / jj_nmc, text=f"Source MC… {m + 1}/{jj_nmc}")
    prog.empty()
    st.session_state["_jjmc"] = dict(
        samples=smp, nominal=float(eng_area), pattern=jj_pat, size=jj_size,
        L=jj_L, n_mc=jj_nmc, mode=params.mode, stack=params.stack, res=res_level,
        angles=jang, angle_labels=_alabels, angle_theta_nom=_athnom)

# Now the engine has run, fill the sidebar Save box with a download button that
# exports every parameter + the junction results (re-loadable via the uploader).
with _save_box:
    st.download_button(
        "💾 Save parameters + results",
        data=_build_export(params, eng, eng_area, eng_ox, eng_oy,
                           eng_njunc, eng_ic, eng_juncs, res_level,
                           combos=eng_combos, jc_al=jc_al, jj_walls=jj_walls),
        file_name="shadowcast_params.json", mime="application/json",
        use_container_width=True)

# ─── Tabs ─────────────────────────────────────────────────────────
(tab1, tab2, tab_play, tab3, tab4, tab_scan, tab_srcmc,
 tab_wafer, tab5) = st.tabs([
    "📐 Cross-section",
    "🗺️ Top View",
    "🎬 Playback",
    "🔄 φ Junction View",
    "🔍 Break Check",
    "📈 Parameter Scan",
    "🎲 Source MC",
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
    st.session_state.setdefault("cs_xc", 0.0)
    st.session_state.setdefault("cs_zmin", 0.0)
    st.session_state["cs_half"] = float(np.clip(st.session_state["cs_half"], 100.0, _gR))
    st.session_state["cs_zmax"] = float(np.clip(st.session_state["cs_zmax"], 100.0, _ztop))
    st.session_state["cs_xc"] = float(np.clip(st.session_state["cs_xc"], -_gR, _gR))
    st.session_state["cs_zmin"] = float(np.clip(st.session_state["cs_zmin"], 0.0, _ztop - 50.0))
    with st.expander("🔍 表示範囲 / 拡大 (View range / zoom)", expanded=False):
        vr1, vr2 = st.columns(2)
        cs_half = vr1.slider("Horizontal half-width [nm]", 100.0, _gR,
                             step=25.0, key="cs_half")
        cs_xc = vr2.slider("Horizontal center [nm]  (pan)", -_gR, _gR,
                           step=25.0, key="cs_xc")
        vr3, vr4 = st.columns(2)
        cs_zmin = vr3.slider("Z min [nm]", 0.0, _ztop - 50.0,
                             step=25.0, key="cs_zmin")
        cs_zmax = vr4.slider("Z max [nm]", 100.0, _ztop,
                             step=25.0, key="cs_zmax")
        st.caption("Narrow the window (and pan with the center slider) to zoom; the "
                   "⤢ fullscreen button on a figure enlarges it.")
    _cs_zmin = min(cs_zmin, cs_zmax - 25.0)               # keep z-window non-inverted

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
                                view_half=cs_half, zmax=cs_zmax,
                                view_center=cs_xc, zmin=_cs_zmin)
        st.pyplot(figs, use_container_width=True)
        plt.close(figs)
        st.markdown("**Combined slice** (all layers, junction highlighted)")
        _show_vox_grid = st.checkbox(
            "Show voxel outlines", key="cs_voxel_grid",
            help="Outline every deposited-metal voxel with a thin white "
                 "line, so the actual voxel size/granularity of the metal "
                 "is visible directly on the image.")
        _has_fine = bool(getattr(eng, "coverage_sub", None))
        _show_fine_detail = st.checkbox(
            "Show fine sub-voxel detail (soft-edge)", key="cs_fine_detail",
            disabled=not _has_fine,
            help="Replace each evaporation's metal near the soft-edge band "
                 "with its true sub-voxel taper shape, correctly stacked "
                 "where one evaporation deposits on top of another (e.g. a "
                 "junction overlap) — instead of the coarse base-grid voxel "
                 "size. Lateral (x-y) sub-sampling makes this finer-grained "
                 "across the slice, one independently-tested column per "
                 "lateral sub-position. Vertical (z) sub-sampling instead "
                 "makes each column's own outermost film boundary more "
                 "precise — shown as a partial-height sliver instead of "
                 "always a whole voxel — rather than adding distinct z "
                 "sub-layers. Columns with more than one disconnected metal "
                 "surface (e.g. sidewall coating plus a separate "
                 "suspended-bridge underside) can't be reconstructed from "
                 "coverage data alone and render exactly as the coarse view "
                 "there instead." +
                 ("" if _has_fine else "  Needs soft edge + lateral/z "
                  "sub-sampling > 1 (no fine data in this result)."))
        if params.stack == "Trilayer":
            _metal_t = {"Evap 1 — Nb": params.tri_t1, "Evap 2 — Al": params.tri_t2,
                       "Evap 3 — Al": params.tri_t3, "Evap 4 — Nb": params.tri_t4}
        else:
            _metal_t = {"Evap 1": params.t_metal1, "Evap 2": params.t_metal2}
        figc = vv.render_cross_section(eng, slice_angle, slice_pos, eng_jm,
                                       view_half=cs_half, zmax=cs_zmax,
                                       view_center=cs_xc, zmin=_cs_zmin,
                                       show_voxel_grid=_show_vox_grid,
                                       fine_detail=_show_fine_detail,
                                       metal_thicknesses=_metal_t)
        st.pyplot(figc, use_container_width=True)
        plt.close(figc)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voxel size", f"{eng.vox:.1f} nm")
    c2.metric("Junction area (engine)", f"{eng_area:.0f} nm²",
              help=("Active area = full 3-D (floor + sidewalls)." if jj_walls
                    else "Active area = horizontal floor overlap only."))
    c3.metric("Overlap x (engine)", f"{eng_ox:.0f} nm")
    c4.metric("Overlap y (engine)", f"{eng_oy:.0f} nm")
    if jj_walls:
        w1, w2, w3 = st.columns(3)
        w1.metric("• Floor area", f"{eng_area_floor:.0f} nm²",
                  help="Horizontal overlap (electrode 2 on top of electrode 1).")
        w2.metric("• Sidewall area", f"{eng_area_walls:.0f} nm²",
                  help="Vertical M-O-M barrier on the resist sidewalls.")
        w3.metric("• Total (floor+walls)", f"{eng_area_full:.0f} nm²",
                  help="Sum used for Ic / R_n while 'Count sidewall' is on.")
        st.caption(
            "Junction area = Al1∩Al2 overlap (Σ cells × voxel²).  **Sidewall "
            "counting ON** → Ic / R_n use the **total** (floor + walls); the "
            "floor-only and sidewall-only parts are split above."
        )
    else:
        st.caption(
            "Junction area = the **true Al1∩Al2 overlap** measured by counting "
            "overlapping voxels (Σ cells × voxel²), so non-rectangular junctions "
            "are handled exactly; overlap x / y are just the bounding-box "
            "extents.  **Floor-only** (vertical sidewall barrier excluded — "
            "enable 'Count sidewall' in the sidebar to add it)."
        )
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Est. critical current Iᶜ", f"{eng_ic:.3f} µA",
              help=f"Al, ~4 K: Jc = {jc_al:.0f} A/cm² (Ambegaokar–Baratoff)")
    e2.metric("Normal resistance R_n", _fmt_rn(eng_jj["Rn_ohm"]),
              help="R_n = πΔ/(2e·Iᶜ) (Ambegaokar–Baratoff, Al Δ≈0.18 meV)")
    e3.metric("Josephson inductance L_J", _fmt_lj(eng_jj["Lj_nH"]),
              help="L_J = ħ / (2e·Iᶜ)")
    e4.metric("Josephson energy E_J/h", f"{eng_jj['Ej_h_GHz']:.2f} GHz",
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
            try:                                  # interactive Plotly (drag-rotate)
                st.plotly_chart(
                    vv.render_thickness_surface_plotly(eng, view_half=top_half),
                    use_container_width=True)
                st.caption("Drag to rotate · scroll to zoom · shift-drag to pan.")
            except Exception:                     # plotly missing → static fallback
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

# ═══ TAB: Playback (step-through deposition) ═════════════════════
with tab_play:
    st.subheader("🎬 Deposition playback — step-through")
    st.markdown(
        "Watch the simulation build up: each evaporation's film grows **layer by "
        "layer** toward the source, then oxidation and lift-off.  Scrub the slider "
        "(or build a GIF).  Uses the **Cross-section** tab's slice angle and view "
        "window.")
    _frames = getattr(eng, "depo_frames", None)
    if not _frames:
        st.info("Press **▶ Run simulation** to generate the deposition timeline.")
    else:
        _nfr = len(_frames)
        st.session_state.setdefault("play_k", _nfr - 1)
        st.session_state["play_k"] = int(np.clip(st.session_state["play_k"],
                                                 0, _nfr - 1))
        if _nfr >= 2:
            k = st.slider("Frame", 0, _nfr - 1, key="play_k",
                          help="0 = bare resist · last = lift-off.  Drag to scrub "
                               "through the deposition.")
        else:
            k = 0
        f = _frames[k]
        st.caption(f"**Step {k + 1} / {_nfr}** — {f['label']}")
        with st.spinner("Rendering frame..."):
            figp = vv.render_deposition_frame(
                eng, f["step"], show_oxide=f["show_oxide"], liftoff=f["liftoff"],
                angle_deg=slice_angle, offset=slice_pos, view_half=cs_half,
                zmax=cs_zmax, view_center=cs_xc, zmin=_cs_zmin,
                title=f"Playback — {f['label']}")
            st.pyplot(figp, use_container_width=True)
            plt.close(figp)
        if st.button("▶ Build animation (GIF)", key="play_gif",
                     help="Render every frame into a downloadable animated GIF."):
            try:
                import io
                from PIL import Image
                imgs = []
                prog = st.progress(0.0, text="Rendering frames…")
                for i, ff in enumerate(_frames):
                    fg = vv.render_deposition_frame(
                        eng, ff["step"], show_oxide=ff["show_oxide"],
                        liftoff=ff["liftoff"], angle_deg=slice_angle,
                        offset=slice_pos, view_half=cs_half, zmax=cs_zmax,
                        view_center=cs_xc, zmin=_cs_zmin,
                        title=f"Playback — {ff['label']}")
                    buf = io.BytesIO(); fg.savefig(buf, format="png", dpi=90)
                    plt.close(fg); buf.seek(0)
                    imgs.append(Image.open(buf).convert("RGB"))
                    prog.progress((i + 1) / _nfr, text=f"Frame {i + 1}/{_nfr}")
                prog.empty()
                gif = io.BytesIO()
                imgs[0].save(gif, format="GIF", save_all=True,
                             append_images=imgs[1:], duration=450, loop=0)
                st.image(gif.getvalue(), caption="Deposition playback")
                st.download_button("💾 Download GIF", data=gif.getvalue(),
                                   file_name="deposition_playback.gif",
                                   mime="image/gif", use_container_width=True)
            except Exception as e:
                st.warning(f"GIF build needs Pillow: {e}")

    if getattr(params, "soft_edge", False):
        st.divider()
        st.subheader("🔬 Soft-edge coverage check")
        st.markdown(
            "Diagnostic for the finite-source penumbra model: the engine computes a "
            "continuous per-cell **coverage fraction** from the source-angle cloud, "
            "then quantises it to an integer voxel-layer count — this reconstructs that "
            "fraction from the deposited floor thickness and shows it as a step plot. "
            "Shaded spans mark the **band** (the cells actually ray-tested against the "
            "source cloud); flat 0/1 regions were never ray-tested. Uses the "
            "**Cross-section** tab's slice angle and perpendicular offset, with its "
            "own independent zoom below.")
        if getattr(eng, "stack", "Bilayer") == "Trilayer":
            _cov_choices = [
                ("Evap 1 — Nb", eng.films["nb1"], params.tri_t1),
                ("Evap 2 — Al", eng.films["al2"], params.tri_t2),
                ("Evap 3 — Al", eng.films["al3"], params.tri_t3),
                ("Evap 4 — Nb", eng.films["nb4"], params.tri_t4),
            ]
        else:
            _cov_choices = [
                ("Evap 1", eng.al1, params.t_metal1),
                ("Evap 2", eng.al2, params.t_metal2),
            ]
        _cov_lbl = st.selectbox("Evaporation", [c[0] for c in _cov_choices],
                                key="cov_evap",
                                help="Which evaporation's floor deposit to inspect.")
        _cov_metal, _cov_tnom = next((m, t) for lbl, m, t in _cov_choices
                                     if lbl == _cov_lbl)
        _cov_n = max(1, round(_cov_tnom / eng.vox))
        _cov_grid = (eng.coverage or {}).get(_cov_lbl)
        _cov_sub = (eng.coverage_sub or {}).get(_cov_lbl)

        _half = np.degrees(np.arctan(
            (params.soft_size / 2.0) / max(params.soft_L, 1e-9)))
        _analytic_w = np.tan(np.radians(_half)) * eng.z_top

        _cov_bands = vv.find_coverage_bands(eng, _cov_metal, _cov_n, coverage_grid=_cov_grid,
                                            coverage_sub=_cov_sub, angle_deg=slice_angle,
                                            offset=slice_pos)

        _ZOOM_MODES = ["Full range (no zoom)", "Fit all bands", "Each band", "Manual"]
        st.session_state.setdefault("cov_zoom_mode", _ZOOM_MODES[1] if _cov_bands else _ZOOM_MODES[0])
        cov_zoom_mode = st.selectbox(
            "Zoom mode", _ZOOM_MODES, key="cov_zoom_mode",
            help="\"Fit all bands\"/\"Each band\" auto-locate the ray-tested "
                 "shadow-edge region(s) for this evaporation along the current "
                 "slice; \"Manual\" uses the sliders below.")

        if cov_zoom_mode == "Each band" and _cov_bands:
            _band_labels = [f"Band {i + 1}  ({c:.0f} nm)" for i, (c, h) in enumerate(_cov_bands)]
            st.session_state["cov_band_idx"] = int(np.clip(
                st.session_state.get("cov_band_idx", 0), 0, len(_cov_bands) - 1))
            st.selectbox("Band", range(len(_cov_bands)), key="cov_band_idx",
                        format_func=lambda i: _band_labels[i])

        if cov_zoom_mode == "Full range (no zoom)" or not _cov_bands:
            cov_half, cov_xc = _gR, 0.0
            if cov_zoom_mode != "Full range (no zoom)" and not _cov_bands:
                st.caption("No soft-edge band found along this slice — showing full range instead.")
        elif cov_zoom_mode == "Fit all bands":
            los = [c - h for c, h in _cov_bands]; his = [c + h for c, h in _cov_bands]
            cov_xc = (min(los) + max(his)) / 2.0
            cov_half = (max(his) - min(los)) / 2.0
        elif cov_zoom_mode == "Each band":
            cov_xc, cov_half = _cov_bands[st.session_state["cov_band_idx"]]
        else:  # Manual
            _cov_half_lo = min(max(2.0, 2.0 * eng.vox), _gR - eng.vox)
            st.session_state.setdefault(
                "cov_half", float(np.clip(3.0 * _analytic_w, _cov_half_lo, cs_half)))
            st.session_state.setdefault("cov_xc", cs_xc)
            st.session_state["cov_half"] = float(
                np.clip(st.session_state["cov_half"], _cov_half_lo, _gR))
            st.session_state["cov_xc"] = float(
                np.clip(st.session_state["cov_xc"], -_gR, _gR))
            with st.expander("🔍 Zoom (band view)", expanded=False):
                cov_half = st.slider("Half-width [nm]", _cov_half_lo, _gR,
                                     step=eng.vox, key="cov_half",
                                     help="Narrow this to frame just the band so you can "
                                          "see whether raising the sub-sampling controls "
                                          "actually resolves finer steps inside it.")
                cov_xc = st.slider("Center [nm]  (pan)", -_gR, _gR,
                                   step=eng.vox, key="cov_xc")
                st.caption("Starts pre-zoomed to ~3× the analytic penumbra-width estimate "
                           "below, centred on the Cross-section tab's current pan; adjust "
                           "freely from there.")

        with st.spinner("Rendering coverage profile..."):
            figc, _band_widths = vv.render_coverage_profile(
                eng, _cov_metal, _cov_n, coverage_grid=_cov_grid, coverage_sub=_cov_sub,
                angle_deg=slice_angle, offset=slice_pos,
                view_half=cov_half, view_center=cov_xc, label=_cov_lbl)
            st.pyplot(figc, use_container_width=True)
            plt.close(figc)
        _measured = (", ".join(f"{w:.0f}" for w in _band_widths)
                    if _band_widths else "—")
        st.caption(
            f"Analytic penumbra-width estimate (source half-angle × resist height) "
            f"≈ {_analytic_w:.0f} nm  •  measured band width(s) in view: "
            f"{_measured} nm  •  voxel = {eng.vox:.1f} nm  "
            f"(need several voxels across the band to resolve the taper — widen the "
            f"source size or raise the grid density if it looks like a single step).")
        if _cov_sub is not None:
            if st.checkbox(
                "Show fine cross-section (lateral sub-voxel detail in the band)",
                key="cov_xsection_show",
                help="Redraws this evaporation's floor metal at the "
                     "soft_supersample_xy resolution instead of the coarse voxel "
                     "grid — only the band (highlighted) is genuinely finer; "
                     "elsewhere a coarse cell's value is just repeated. "
                     "Diagnostic only: each fine column is a flat vertical "
                     "stack from the floor, not the engine's true growth path."):
                with st.spinner("Rendering fine cross-section..."):
                    figx = vv.render_coverage_cross_section(
                        eng, _cov_metal, _cov_n, _cov_sub,
                        angle_deg=slice_angle, offset=slice_pos,
                        view_half=cov_half, view_center=cov_xc, label=_cov_lbl)
                    st.pyplot(figx, use_container_width=True)
                    plt.close(figx)

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
        ic = area_nm2 * ic_factor
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
                p.tri_angle2, p.tri_phi2, p.tri_angle4, p.tri_phi4,
                getattr(p, "sidewall", False), getattr(p, "resist_round", 0.0),
                getattr(p, "resist_round_method", "analytic"),
                getattr(p, "soft_edge", False), getattr(p, "soft_pattern", "rotline"),
            getattr(p, "soft_size", 12.0), getattr(p, "soft_L", 550.0),
            getattr(p, "soft_rays", 24), getattr(p, "soft_supersample_xy", 1),
            getattr(p, "soft_supersample_z", 1))

    @st.cache_data(show_spinner=False)
    def _scan_area(sig, _p):
        r = simulate(_p, max_cells=_smc, min_vox=_smv)
        _, a_full, _, _, _ = junction_footprint(r, include_sidewalls=True)
        _, a_floor, _, _, _ = junction_footprint(r, include_sidewalls=False)
        return float(a_floor), float(a_full)

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
                    _af, _afull = _scan_area(_scan_sig(p2), p2)
                    areas[i] = _afull if jj_walls else _af
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
                    _af, _afull = _scan_area(_scan_sig(p2), p2)
                    areas[iy, ix] = _afull if jj_walls else _af
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

# ═══ TAB: Source MC (finite beam-pattern e-beam source — single JJ) ══
with tab_srcmc:
    st.subheader("Finite e-beam source — JJ-area Monte-Carlo")
    st.caption("Models the e-beam source with a finite spatial spread set by its "
               "raster pattern (rotating line / uniform disk / Gaussian / point) at "
               "throw distance L, instead of an ideal point. Each run independently "
               "jitters every evaporation's beam angle, giving the probabilistic "
               "JJ-area distribution. Enable it and press ▶ Run in the left sidebar.")
    mc = st.session_state.get("_jjmc")
    if not st.session_state.get("jj_src"):
        st.info("Enable **🎲 Finite e-beam source** in the left sidebar, choose a "
                "beam pattern / size / samples, then press ▶ Run source Monte-Carlo.")
    elif mc is None:
        st.info("Press **▶ Run source Monte-Carlo** in the sidebar to compute the "
                "JJ-area distribution.")
    else:
        s = mc["samples"]
        mean = float(np.mean(s)); std = float(np.std(s))
        rel = (std / mean * 100.0) if mean > 0 else float("nan")
        nominal = mc["nominal"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean JJ area", f"{mean:,.0f} nm²")
        c2.metric("σ (1σ error)", f"{std:,.0f} nm²", f"{rel:.2f}% rel")
        c3.metric("Point-source area", f"{nominal:,.0f} nm²",
                  (f"{(mean - nominal) / nominal * 100:+.2f}% vs mean")
                  if nominal > 0 else None)
        c4.metric("MC samples", f"{mc['n_mc']}")
        # Source-plane intensity distribution (2-D scatter + radial profile).
        _sc, _rad = _source_dist_charts(mc["pattern"], mc["size"])
        st.altair_chart(_sc | _rad, use_container_width=False)
        # JJ-area distribution from the Monte-Carlo.
        gmin, gmax = float(np.min(s)), float(np.max(s))
        if gmax <= gmin:
            gmax = gmin + 1.0
        base = alt.Chart(pd.DataFrame({"area": s})).mark_bar(color="#64B5F6").encode(
            x=alt.X("area:Q", bin=alt.Bin(extent=[gmin, gmax], maxbins=24),
                    title="JJ area [nm²]"),
            y=alt.Y("count()", title="samples"))
        r_mean = alt.Chart(pd.DataFrame({"v": [mean]})).mark_rule(
            color="#CE93D8", size=2).encode(x="v:Q")
        r_nom = alt.Chart(pd.DataFrame({"v": [nominal]})).mark_rule(
            color="#FFB74D", strokeDash=[6, 4], size=2).encode(x="v:Q")
        st.altair_chart((base + r_mean + r_nom).properties(
            height=320, title="JJ-area distribution (Monte-Carlo)"),
            use_container_width=True)
        _sz = (f"σ = {mc['size']:.1f} mm" if mc["pattern"] == "gaussian"
               else f"diameter = {mc['size']:.1f} mm")
        st.caption(f"Pattern: {mc['pattern']} ({_sz}), L = {mc['L']:.0f} mm • "
                   f"N_mc = {mc['n_mc']} • {mc['res']}.  Top: source-plane sample "
                   f"cloud + radial profile P(ρ).  Bottom: JJ-area histogram — "
                   f"purple line = MC mean, dashed orange = point-source area.")
        # Per-evaporation beam-angle distributions & correlations.
        if mc.get("angles") is not None:
            with st.expander("📐 Beam-angle distributions & correlations",
                             expanded=False):
                _angle_distribution_ui("jjang", mc["angles"], mc["angle_labels"],
                                       mc["angle_theta_nom"], area=mc["samples"])

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

    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    waf_L = wc1.number_input(
        "Throw distance L [mm]", min_value=20.0, max_value=2000.0,
        value=550.0, step=10.0, key="waf_L",
        help="Source→wafer-centre distance. Smaller L ⇒ stronger position "
             "dependence (deviation ≈ r/L).")
    waf_size = wc2.selectbox(
        "Wafer size", list(WAFER_SPECS.keys()), index=2, key="waf_size",
        help="Real wafer disk with its primary flat (オリフラ) at the bottom; "
             "off-wafer grid cells are skipped.")
    R_mm, c_flat = WAFER_SPECS[waf_size]
    d_flat = float(np.sqrt(R_mm ** 2 - (c_flat / 2.0) ** 2))
    # Cell size first (wc4) so the Grid-N max can scale with it.  st.columns
    # placement is independent of call order, so N still renders in column 3.
    waf_cell = float(wc4.number_input(
        "Grid cell size [mm]", min_value=1.0, max_value=10.0, value=5.0, step=0.1,
        key="waf_cell",
        help="Pitch of the N×N grid, centred on the wafer; a small N need not "
             "reach the edge."))
    # Max N = the grid that just spans the full wafer diameter (2R) at this
    # pitch; beyond it only adds off-wafer corner cells.  Capped for sanity
    # (each grid cell is a full engine run).
    N_FILL_CAP = 41
    n_fill = int(np.ceil(2.0 * R_mm / waf_cell)) + 1
    waf_n_max = max(2, min(n_fill, N_FILL_CAP))
    if int(st.session_state.get("waf_n", 5)) > waf_n_max:
        st.session_state["waf_n"] = waf_n_max        # avoid Streamlit max error
    waf_n = int(wc3.number_input(
        "Grid N (N×N)", min_value=2, max_value=waf_n_max, value=5, step=1,
        key="waf_n",
        help=f"Cells per side of the centred N×N grid.  Max {waf_n_max} = the N "
             f"whose {waf_cell:.1f} mm pitch spans the full {waf_size} wafer "
             f"(2R = {2 * R_mm:.0f} mm); raise the cell size to fill the wafer "
             f"with fewer cells."))
    waf_res = wc5.selectbox(
        "Voxel grid density", list(RES_LEVELS.keys()), key="waf_res",
        help="Finer = slower; each grid cell is a full engine run.")
    _wsmc, _wsmv = RES_LEVELS[waf_res]
    # Centred N×N grid with the chosen pitch (fills from the centre; cells outside
    # the disk or below the primary flat are skipped).  Reused below.
    wcoords = (np.arange(waf_n) - (waf_n - 1) / 2.0) * waf_cell
    Xg, Yg = np.meshgrid(wcoords, wcoords)
    wmask = (Xg ** 2 + Yg ** 2 <= R_mm ** 2) & (Yg >= -d_flat)
    n_on = int(wmask.sum())
    _span = (waf_n - 1) * waf_cell
    _fills = _span >= 2 * R_mm
    st.caption(f"{waf_size} wafer (R = {R_mm:.1f} mm) • {waf_n}×{waf_n} grid at "
               f"{waf_cell:.1f} mm pitch (span {_span:.1f} mm, centred) • "
               f"{n_on}/{waf_n * waf_n} on-wafer cells at '{waf_res}'"
               + (" • spans full wafer" if _fills else "") + ".")
    if n_on > 200:
        st.caption(f"⚠ {n_on} on-wafer cells = {n_on} full engine runs — this "
                   f"may take a while.")

    # Finite source — opt-in beam-pattern Monte-Carlo JJ-area statistics.
    gauss_on = st.checkbox(
        "Finite source — beam-pattern Monte-Carlo JJ-area statistics",
        key="waf_gauss",
        help="Model the source with a finite spatial spread set by its raster "
             "pattern (rotating line / uniform disk / Gaussian / point) instead of an "
             "ideal point.  The engine runs N_mc times per on-wafer cell to build a "
             "per-cell mean ± σ of the JJ area.")
    waf_pat, waf_size_src, n_mc = "point", 0.0, 0
    if gauss_on:
        gc1, gc2 = st.columns(2)
        waf_pat, waf_size_src = _beam_pattern_controls("waf_src", gc1)
        n_mc = int(gc2.number_input(
            "Monte-Carlo samples / cell", min_value=2, max_value=200, value=20,
            step=1, key="waf_nmc",
            help="Engine runs per on-wafer cell.  Total = N_mc × on-wafer "
                 "cells — keep modest; this is the expensive path."))
        _wspread = (waf_size_src if waf_pat == "gaussian"
                    else waf_size_src / 2.0) / waf_L
        st.caption(
            f"≈ {n_mc * n_on} engine runs (N_mc={n_mc} × {n_on} cells) • pattern "
            f"'{waf_pat}' • spread ≈ {np.degrees(_wspread):.3f}° at L={waf_L:.0f} mm.")
        # Source-plane intensity distribution (2-D scatter + radial profile).
        _wsc, _wrad = _source_dist_charts(waf_pat, waf_size_src)
        st.altair_chart(_wsc | _wrad, use_container_width=False)

    # Per-evaporation schematic of the fixed source + tilted wafer (always on).
    with st.spinner("Drawing source / wafer-tilt schematic…"):
        figgeo = vv.render_wafer_geometry(params, waf_L, c_flat / R_mm)
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
                _wsmc, _wsmv, getattr(p, "sidewall", False),
                getattr(p, "resist_round", 0.0),
                getattr(p, "resist_round_method", "analytic"),
                getattr(p, "soft_edge", False), getattr(p, "soft_pattern", "rotline"),
            getattr(p, "soft_size", 12.0), getattr(p, "soft_L", 550.0),
            getattr(p, "soft_rays", 24), getattr(p, "soft_supersample_xy", 1),
            getattr(p, "soft_supersample_z", 1))

    @st.cache_data(show_spinner=False)
    def _wafer_area(sig, _p):
        r = simulate(_p, max_cells=_wsmc, min_vox=_wsmv)
        _, a_full, ox, oy, _ = junction_footprint(r, include_sidewalls=True)
        _, a_floor, _, _, _ = junction_footprint(r, include_sidewalls=False)
        return float(a_floor), float(a_full), float(ox), float(oy)

    if st.button("▶ Run wafer map", key="run_wafer",
                 use_container_width=True):
        if n_on == 0:
            st.warning("No grid cell lands on the wafer at this N — "
                       "increase **Grid N (N×N)**.")
        else:
            # Per-evaporation effective (θ′, φ′) over the whole grid — pure math,
            # vectorised, computed once (the engine still runs per on-wafer cell).
            theta_grids, phi_grids, dist_grids = {}, {}, {}
            for lbl, _ta, _pa, th, ph in evap_beams(params):
                tg, pg = wafer_local_angles(th, ph, Xg, Yg, waf_L)
                theta_grids[lbl], phi_grids[lbl] = tg, pg
                dist_grids[lbl] = wafer_source_dist(th, ph, Xg, Yg, waf_L)

            warea = np.full((waf_n, waf_n), np.nan)       # [iy, ix]; off-wafer NaN
            wox = np.full((waf_n, waf_n), np.nan)         # per-cell overlap x [nm]
            woy = np.full((waf_n, waf_n), np.nan)         # per-cell overlap y [nm]
            wstd = wsamp = wang = None
            _walabels, _wathnom = _beam_angle_meta(params)
            if gauss_on:
                # Finite-source Monte-Carlo: N_mc engine runs per on-wafer cell.
                # A fixed seed makes the result reproducible and cache-friendly
                # on re-press (each perturbed-angle sample is a distinct
                # _wafer_area cache key).
                wstd = np.full((waf_n, waf_n), np.nan)
                wsamp = np.full((waf_n, waf_n, n_mc), np.nan)   # raw MC areas
                wang = np.full((waf_n, waf_n, n_mc, len(_walabels)), np.nan)
                rng = np.random.default_rng(0)
                done, total = 0, n_on * n_mc
                prog = st.progress(0.0, text="Monte-Carlo sampling…")
                for iy in range(waf_n):
                    for ix in range(waf_n):
                        if not wmask[iy, ix]:
                            continue                      # skip off-wafer cells
                        smp = np.empty(n_mc)
                        smp_ox = np.empty(n_mc); smp_oy = np.empty(n_mc)
                        for m in range(n_mc):
                            q = wafer_params_source(
                                params, float(Xg[iy, ix]), float(Yg[iy, ix]),
                                waf_L, waf_pat, waf_size_src, rng)
                            _af, _afull, _ox, _oy = _wafer_area(_wafer_sig(q), q)
                            smp[m] = _afull if jj_walls else _af
                            smp_ox[m] = _ox; smp_oy[m] = _oy
                            wang[iy, ix, m, :] = _beam_angle_row(q, params)
                            done += 1
                            prog.progress(done / total,
                                          text=f"Monte-Carlo… {done}/{total}")
                        warea[iy, ix] = float(np.mean(smp))
                        wox[iy, ix] = float(np.mean(smp_ox))
                        woy[iy, ix] = float(np.mean(smp_oy))
                        wstd[iy, ix] = float(np.std(smp))
                        wsamp[iy, ix, :] = smp
                prog.empty()
            else:
                done = 0
                prog = st.progress(0.0, text="Simulating wafer positions…")
                for iy in range(waf_n):
                    for ix in range(waf_n):
                        if not wmask[iy, ix]:
                            continue                      # skip off-wafer cells
                        q = wafer_params(params, float(Xg[iy, ix]),
                                         float(Yg[iy, ix]), waf_L)
                        _af, _afull, _ox, _oy = _wafer_area(_wafer_sig(q), q)
                        warea[iy, ix] = _afull if jj_walls else _af
                        wox[iy, ix] = _ox; woy[iy, ix] = _oy
                        done += 1
                        prog.progress(done / n_on,
                                      text=f"Simulating… {done}/{n_on}")
                prog.empty()
            st.session_state["_wafermap"] = dict(
                coords=wcoords, areas=warea, ox=wox, oy=woy,
                std=wstd, samples=wsamp,
                theta=theta_grids, phi=phi_grids, dist=dist_grids,
                L=waf_L, R=R_mm, c=c_flat, d=d_flat,
                size=waf_size, n=waf_n, res=waf_res, mode=params.mode,
                stack=params.stack, gauss=bool(gauss_on), pattern=waf_pat,
                src_size=waf_size_src, n_mc=n_mc, sidewall=bool(params.sidewall),
                walls=bool(jj_walls),
                angles=wang, angle_labels=_walabels, angle_theta_nom=_wathnom)

    wm = st.session_state.get("_wafermap")
    if wm and "theta" in wm:                              # new-format result
        warea = wm["areas"]
        wstd = wm.get("std")                              # per-cell 1σ (or None)
        gauss = bool(wm.get("gauss"))                     # finite-source MC?

        # (a) Drawn wafer (matplotlib): area + Ic clipped to the disk + flat.
        figw = vv.render_wafer_map_2d(
            wm["coords"], warea, warea * ic_factor, wm["R"], wm["c"], wm["d"],
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
        _dist = wm.get("dist", {})                        # per-evap source distance
        _rn = _rn_from_ic_uA(warea[on] * ic_factor)        # R_n [Ω] per cell
        _rn = np.where(np.isfinite(_rn), _rn, np.nan)     # ∞ (open) → null for Vega

        # Selectable 2D-map quantity (colour + optional in-cell label).  Each
        # entry: display-name → 2-D grid over the whole N×N (colour map is global).
        _rn2d = _rn_from_ic_uA(warea * ic_factor)
        _rn2d = np.where(np.isfinite(_rn2d), _rn2d, np.nan)
        metrics = {"JJ area [nm²]": warea,
                   "Est. Ic [µA]":  warea * ic_factor,
                   "R_n [kΩ]":      _rn2d / 1000.0}
        _wox, _woy = wm.get("ox"), wm.get("oy")           # None for legacy results
        if _wox is not None:
            metrics["Overlap x [nm]"] = _wox
        if _woy is not None:
            metrics["Overlap y [nm]"] = _woy
        for lbl in wm["theta"]:
            metrics[f"{lbl} θ′ [°]"] = wm["theta"][lbl]
            metrics[f"{lbl} φ′ [°]"] = wm["phi"][lbl]
            if lbl in _dist:
                metrics[f"{lbl} dist [mm]"] = _dist[lbl]
        if gauss:
            metrics["area σ [nm²]"] = wstd
        COLORMAPS = {
            "Green→Yellow→Red": ("redyellowgreen", True),   # green=low … red=high
            "Viridis": ("viridis", False), "Magma": ("magma", False),
            "Plasma": ("plasma", False),   "Turbo": ("turbo", False),
            "Blue→Red": ("redblue", True), "Cividis": ("cividis", False)}
        cc1, cc2, cc3, cc4 = st.columns([2, 2, 1, 2])
        msel = cc1.selectbox("2D map value", list(metrics), key="wmap_metric")
        cmsel = cc2.selectbox("Colour map", list(COLORMAPS), key="wmap_cmap")
        sig = int(cc3.number_input("Sig figs", 1, 5, 2, 1, key="wmap_sig"))
        show_vals = cc4.checkbox("Show value in each cell", key="wmap_labels")
        _scheme, _rev = COLORMAPS[cmsel]
        _mval_on = np.asarray(metrics[msel])[on].astype(float)
        _mval_on = np.where(np.isfinite(_mval_on), _mval_on, np.nan)

        cdata = {"x": Xg[on], "y": Yg[on], "area": warea[on],
                 "Ic": warea[on] * ic_factor, "Rn": _rn / 1000.0,
                 "val": _mval_on, "label": [_fmt_sig(v, sig) for v in _mval_on]}
        tips = [alt.Tooltip("x:Q", title="x [mm]", format=".1f"),
                alt.Tooltip("y:Q", title="y [mm]", format=".1f"),
                alt.Tooltip("area:Q", title="area [nm²]", format=".0f"),
                alt.Tooltip("Ic:Q", title="Ic [µA]", format=".4f"),
                alt.Tooltip("Rn:Q", title="R_n [kΩ]", format=".3f")]
        for lbl in wm["theta"]:
            key = lbl.replace(" ", "_")
            cdata[f"{key}_th"] = wm["theta"][lbl][on]
            cdata[f"{key}_ph"] = wm["phi"][lbl][on]
            tips.append(alt.Tooltip(f"{key}_th:Q", title=f"{lbl} θ′ [°]",
                                    format=".2f"))
            tips.append(alt.Tooltip(f"{key}_ph:Q", title=f"{lbl} φ′ [°]",
                                    format=".2f"))
            if lbl in _dist:                              # source→device distance
                cdata[f"{key}_d"] = _dist[lbl][on]
                tips.append(alt.Tooltip(f"{key}_d:Q", title=f"{lbl} dist [mm]",
                                        format=".1f"))
        if gauss:                                         # finite-source error
            cdata["area_std"] = wstd[on]
            with np.errstate(divide="ignore", invalid="ignore"):
                cdata["rel"] = np.where(warea[on] != 0,
                                        wstd[on] / warea[on] * 100.0, np.nan)
            tips.append(alt.Tooltip("area_std:Q", title="area σ [nm²]",
                                    format=".0f"))
            tips.append(alt.Tooltip("rel:Q", title="rel err [%]", format=".2f"))
        cdf = pd.DataFrame(cdata)
        cdf = cdf.assign(x0=cdf["x"] - h / 2, x1=cdf["x"] + h / 2,
                         y0=cdf["y"] - h / 2, y1=cdf["y"] + h / 2,
                         cid=np.arange(len(cdf)))         # cell id (on-wafer order)
        Rd = wm["R"] * 1.05                              # equal, symmetric domain
        heat = alt.Chart(cdf).mark_rect().encode(
            x=alt.X("x0:Q", title="wafer x  [mm]",
                    scale=alt.Scale(domain=[-Rd, Rd], nice=False)), x2="x1:Q",
            y=alt.Y("y0:Q", title="wafer y  [mm]",
                    scale=alt.Scale(domain=[-Rd, Rd], nice=False)), y2="y1:Q",
            color=alt.Color("val:Q", title=msel,
                            scale=alt.Scale(scheme=_scheme, reverse=_rev),
                            legend=alt.Legend(orient="bottom")),
            tooltip=tips)
        tt = np.linspace(np.arctan2(-wm["d"], wm["c"] / 2),
                         np.arctan2(-wm["d"], -wm["c"] / 2) + 2 * np.pi, 200)
        out = pd.DataFrame({
            "x": np.r_[wm["R"] * np.cos(tt), -wm["c"] / 2, wm["c"] / 2],
            "y": np.r_[wm["R"] * np.sin(tt), -wm["d"], -wm["d"]],
            "o": np.arange(202)})
        outline = alt.Chart(out).mark_line(color="#90A4AE").encode(
            x="x:Q", y="y:Q", order="o:O")
        _layers = heat + outline
        if show_vals:                                     # in-cell value labels
            _mid = float(np.nanmedian(_mval_on)) if np.isfinite(_mval_on).any() \
                else 0.0
            text = alt.Chart(cdf).mark_text(fontSize=9, baseline="middle").encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[-Rd, Rd], nice=False)),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[-Rd, Rd], nice=False)),
                text="label:N",
                color=alt.condition(alt.datum.val > _mid, alt.value("black"),
                                    alt.value("white")))
            _layers = _layers + text
        st.altair_chart(_layers.properties(width=460, height=460),
                        use_container_width=False)
        st.caption(f"Colour = **{msel}** ({cmsel}).  "
                   + (f"In-cell values at {sig} sig figs.  " if show_vals else "")
                   + "Hover any cell for its area, Ic, R_n and per-evaporation "
                     "effective (θ′, φ′).")
        if gauss:                                         # σ (1σ error) heatmap
            heat_std = alt.Chart(cdf).mark_rect().encode(
                x=alt.X("x0:Q", title="wafer x  [mm]",
                        scale=alt.Scale(domain=[-Rd, Rd], nice=False)),
                x2="x1:Q",
                y=alt.Y("y0:Q", title="wafer y  [mm]",
                        scale=alt.Scale(domain=[-Rd, Rd], nice=False)),
                y2="y1:Q",
                color=alt.Color("area_std:Q", title="area σ [nm²]",
                                scale=alt.Scale(scheme="magma"),
                                legend=alt.Legend(orient="bottom")),
                tooltip=tips)
            samp = wm.get("samples")
            if samp is None:                              # legacy dict; σ map only
                st.altair_chart(
                    (heat_std + outline).properties(
                        width=460, height=460,
                        title="JJ-area σ (1σ error) from finite-source "
                              "Monte-Carlo"),
                    use_container_width=False)
                st.caption(f"Per-cell 1σ over N_mc={wm.get('n_mc')} samples "
                           f"(source: {wm.get('pattern', 'gaussian')}, "
                           f"{wm.get('src_size', wm.get('sigma', 0.0)):.1f} mm).")
            else:
                # Hover a cell on the σ map → its JJ-area distribution (right).
                samp_on = samp[on]                        # (n_on, n_mc)
                n_cells, nmc = samp_on.shape
                long_df = pd.DataFrame({
                    "cid": np.repeat(np.arange(n_cells), nmc),
                    "area": samp_on.reshape(-1)})
                center_cid = int(np.argmin(Xg[on] ** 2 + Yg[on] ** 2))
                csel = alt.selection_point(
                    fields=["cid"], on="mouseover", empty=False,
                    value=[{"cid": center_cid}])
                gmin = float(np.nanmin(samp_on))
                gmax = float(np.nanmax(samp_on))
                if gmax <= gmin:                          # all samples identical
                    gmax = gmin + 1.0
                hist = alt.Chart(long_df).mark_bar(color="#64B5F6").encode(
                    x=alt.X("area:Q",
                            bin=alt.Bin(extent=[gmin, gmax], maxbins=18),
                            title="JJ area [nm²]"),
                    y=alt.Y("count()", title="samples")
                    ).transform_filter(csel).properties(
                        width=360, height=480,
                        title="Hovered-cell JJ-area distribution")
                sig_map = (heat_std.add_params(csel) + outline).properties(
                    width=460, height=460,
                    title="JJ-area σ (1σ error) — hover a cell for its "
                          "distribution")
                st.altair_chart(sig_map | hist, use_container_width=False)
                st.caption(
                    f"Left: per-cell 1σ over N_mc={wm.get('n_mc')} samples "
                    f"(source: {wm.get('pattern', 'gaussian')}, "
                    f"{wm.get('src_size', wm.get('sigma', 0.0)):.1f} mm).  Right: the "
                    f"hovered cell's JJ-area histogram (bin extent shared "
                    f"across cells; defaults to the centre cell).")

            # Per-cell beam-angle distributions & correlations.
            if wm.get("angles") is not None:
                st.markdown("**Per-cell beam-angle distributions & correlations**")
                ij = list(zip(*np.where(on)))     # (iy,ix) in C-order == Xg[on]
                xs_on, ys_on = Xg[on], Yg[on]
                cell_labels = [f"({xs_on[k]:.0f}, {ys_on[k]:.0f}) mm"
                               for k in range(len(ij))]
                cc = int(np.argmin(xs_on ** 2 + ys_on ** 2))   # centre cell default
                k = st.selectbox("Cell", range(len(ij)), index=cc,
                                 format_func=lambda i: cell_labels[i],
                                 key="wafang_cell")
                iy, ix = ij[k]
                _angle_distribution_ui("wafang", wm["angles"][iy, ix],
                                       wm["angle_labels"], wm["angle_theta_nom"],
                                       area=wm["samples"][iy, ix])
                st.caption(f"Cell ({xs_on[k]:.1f}, {ys_on[k]:.1f}) mm — beam angles "
                           f"over N_mc={wm.get('n_mc')} finite-source draws.")

        # (c) Sortable table of every on-wafer cell (pretty column names).
        rows = {"x [mm]": Xg[on].round(2), "y [mm]": Yg[on].round(2),
                "area [nm²]": warea[on].round(0),
                "Ic [µA]": (warea[on] * ic_factor).round(4),
                "R_n [kΩ]": (_rn_from_ic_uA(warea[on] * ic_factor) / 1000.0).round(3)}
        if gauss:
            rows["area σ [nm²]"] = wstd[on].round(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rows["rel err [%]"] = np.where(
                    warea[on] != 0, wstd[on] / warea[on] * 100.0,
                    np.nan).round(2)
        for lbl in wm["theta"]:
            rows[f"{lbl} θ′ [°]"] = wm["theta"][lbl][on].round(2)
            rows[f"{lbl} φ′ [°]"] = wm["phi"][lbl][on].round(2)
            if lbl in _dist:
                rows[f"{lbl} dist [mm]"] = _dist[lbl][on].round(1)
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
        wic = warea * ic_factor
        _rn_on = _rn_from_ic_uA(warea[on] * ic_factor)
        _rn_fin = _rn_on[np.isfinite(_rn_on)]
        _rn_txt = (f"{_fmt_rn(float(_rn_fin.min()))}–{_fmt_rn(float(_rn_fin.max()))}"
                   if _rn_fin.size else "—")
        st.caption(
            f"Centre cell (nominal single-JJ) = {warea[ci, ci]:.0f} nm²  •  "
            f"std/mean = {cv:.1f} %  •  Ic range "
            f"{np.nanmin(wic):.3f}–{np.nanmax(wic):.3f} µA  •  R_n range {_rn_txt}.  "
            f"JJ area = **{'floor + sidewalls' if wm.get('walls') else 'floor only'}**.")
        if wm.get("sidewall"):
            st.caption("⚙ Side-wall effect ON — the 1st-evaporation wall coating "
                       "narrows later evaporations; the narrowing grows with the "
                       "local incident angle, so the area / R_n map is asymmetric "
                       "across the wafer (Jpn. J. Appl. Phys. aca256).")
        if gauss:
            mean_std = float(np.nanmean(wstd))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_map = np.where(warea != 0, wstd / warea * 100.0, np.nan)
            max_rel = float(np.nanmax(rel_map))
            g1, g2, g3 = st.columns(3)
            g1.metric("Mean σ (error)", f"{mean_std:.0f} nm²",
                      help="Mean over on-wafer cells of the per-cell 1σ JJ-area "
                           "spread from the finite-source Monte-Carlo.")
            g2.metric("Max rel error", f"{max_rel:.2f} %",
                      help="Largest per-cell σ/mean across the wafer.")
            g3.metric("MC samples / cell", f"{wm.get('n_mc')}")
            _wp = wm.get('pattern', 'gaussian')
            _wsz = float(wm.get('src_size', wm.get('sigma', 0.0)))
            _wsp = (_wsz if _wp == 'gaussian' else _wsz / 2.0) / wm['L']
            st.caption(
                f"Finite source — pattern '{_wp}', size {_wsz:.1f} mm "
                f"(angular spread ≈ {np.degrees(_wsp):.3f}° at L={wm['L']:.0f} mm).  "
                f"‘Error’ = per-cell 1σ of the JJ area over N_mc={wm.get('n_mc')} "
                f"Monte-Carlo depositions.")
    else:
        st.info("Set the source/grid configuration above and press "
                "**Run wafer map**.")

# ═══ TAB 5: Junction Area ════════════════════════════════════════
with tab5:
    st.subheader("Junction Area & Full Parameter Summary")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Overlap x (engine)",  f"{eng_ox:.0f} nm")
    c2.metric("Overlap y (engine)",  f"{eng_oy:.0f} nm")
    c3.metric("Area A (engine)",     f"{eng_area:.0f} nm²",
              help=("Active = floor + sidewalls." if jj_walls
                    else "Active = floor only."))
    c4.metric("Est. Ic (engine)",    f"{eng_ic:.3f} µA",
              help=f"Al, 4K: Jc = {jc_al:.0f} A/cm² (Ambegaokar-Baratoff)")
    c5.metric("Junctions",           f"{eng_njunc}",
              help="Number of spatially separate Al1∩Al2 overlaps")
    if jj_walls:
        w1, w2, w3 = st.columns(3)
        w1.metric("• Floor area",  f"{eng_area_floor:.0f} nm²")
        w2.metric("• Sidewall area", f"{eng_area_walls:.0f} nm²")
        w3.metric("• Total (floor+walls)", f"{eng_area_full:.0f} nm²",
                  help="Ic / R_n use this total while 'Count sidewall' is on.")
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Normal resistance R_n", _fmt_rn(eng_jj["Rn_ohm"]),
              help="R_n = πΔ/(2e·Iᶜ) (Ambegaokar–Baratoff, Al Δ≈0.18 meV)")
    j2.metric("Josephson inductance L_J", _fmt_lj(eng_jj["Lj_nH"]),
              help="L_J = ħ / (2e·Iᶜ)")
    j3.metric("Josephson energy E_J/h", f"{eng_jj['Ej_h_GHz']:.2f} GHz",
              help="E_J = (Φ₀/2π)·Iᶜ = ħ·Iᶜ/2e")
    j4.metric("E_J / k_B", f"{eng_jj['Ej_kB_K']:.2f} K")
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
                           combos=eng_combos, jc_al=jc_al, jj_walls=jj_walls),
        file_name="shadowcast_params.json", mime="application/json",
        help="Re-load this file with the sidebar uploader to restore every "
             "parameter.")
