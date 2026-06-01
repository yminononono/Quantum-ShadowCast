"""
manhattan_check.py  (v4)
Cross-shaped Manhattan: Evap1 φ≈0, Evap2 φ≈90°
"""
from process_engine import ProcessParams
from junction_area import compute_junction_area


def manhattan_break_check(params: ProcessParams) -> dict:
    res = compute_junction_area(params)
    ox  = res["overlap_x_nm"]
    oy  = res["overlap_y_nm"]

    if ox <= 0 and oy <= 0:
        status = "❌ Open circuit (both axes)"
    elif ox <= 0:
        status = "❌ Open circuit (x-axis)"
    elif oy <= 0:
        status = "❌ Open circuit (y-axis)"
    elif min(ox, oy) < 20:
        status = "⚠️ Marginal overlap"
    else:
        status = "✅ Junction formed"

    return dict(
        litho_gap_nm=params.bridge_width,
        undercut_nm=params.undercut,
        overlap_x_nm=ox, overlap_y_nm=oy,
        effective_gap_nm=min(ox, oy),
        status=status,
        area_nm2=res["area_nm2"],
        junction_tilt_deg=res["junction_tilt_deg"],
        ic_estimate_uA=res["ic_estimate_uA"],
    )
