# ShadowCast — Josephson Junction Shadow-Evaporation Simulator

A Streamlit app that simulates the fabrication of **Josephson junctions** by
**shadow / oblique evaporation** in 3D. It supports both the **Dolan bridge**
and **Manhattan** geometries, and computes — from the resist profile and
evaporation angles — the junction overlap area, critical current, inductance,
and Josephson energy.

The physics core is a 3D voxel ray-cast engine (`deposition3d.py`): it
ray-traces the tilted evaporation beams into a voxel grid to find where each
metal film is deposited, then extracts the junction as the region where the
first and second metal films overlap across the oxide barrier. This 3D engine
is the **source of truth** — all on-screen views and judgments are based on it.

---

## Features

- **Two fabrication modes**
  - **Dolan bridge**: uniaxial tilt (same φ, opposite θ) so the deposition wraps
    under the suspended bridge.
  - **Manhattan**: two evaporation beams (with independently configurable
    θ₁/φ₁ and θ₂/φ₂) that cross each other.
- **3D shadow-evaporation engine**: voxel ray-casting that reproduces the metal
  films, the oxide, and the resist undercut.
- **Arbitrary junction shape**: the junction is not assumed to be square — it is
  the true overlap of film 1 ∩ film 2 (even when non-rectangular), measured by
  cell count.
- **Electrical quantities** derived from the junction area:
  - Critical current Ic (Ambegaokar–Baratoff, jc = 10 kA/cm²)
  - Josephson inductance L_J = ħ / (2e·Ic)
  - Josephson energy E_J = (Φ₀/2π)·Ic (also shown as E_J/h [GHz] and E_J/k_B [K])
- **Six visualization tabs**
  1. **📐 Cross-section** — cross-section that can be rotated to any in-plane
     angle and offset (with evaporation-beam arrows)
  2. **🗺️ Top View** — top view (metal films, shadow, undercut, junction region)
  3. **🔄 φ Junction View** — zoomed top view around the junction
  4. **🔍 Break Check** — open/short verdict and electrical metrics
  5. **📈 Parameter Scan** — parameter sweep (see below)
  6. **📊 Junction Area** — full parameter summary and result export
- **Parameter Scan**
  - **1D / 2D** sweeps (2D rendered as a heatmap)
  - The **value range and number of points** are configurable per variable
  - The sweep **voxel density (resolution)** offers the same 5 presets as the
    sidebar
  - The output plots **junction area, Ic, L_J, E_J/h, and E_J/k_B all stacked
    vertically**
  - For Manhattan, the per-beam **θ₁ / φ₁ / θ₂ / φ₂** are also sweepable
- **Save / load parameters**: store and restore settings as JSON, plus a
  reset-to-defaults button
- **GDS import** (optional): reads GDSII layout files (requires `gdstk`)

---

## Requirements

- **Python 3.10+** (developed and tested on 3.11)
- Core packages: `streamlit`, `numpy`, `matplotlib`
- Only for GDS import: `gdstk`

Dependencies are kept minimal (`scipy` / `rtree` are **not** required).

---

## Installation

```bash
# 1. Clone and move into the repository
git clone <repository-url>
cd <repository>

# 2. (Recommended) create a conda environment
conda create -n shadowcast python=3.11
conda activate shadowcast

# 3. Install dependencies
pip install -r requirements.txt
```

The minimal set in `requirements.txt` is enough to run the app:

```
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
gdstk>=0.9.0          # only if you use GDS import
```

---

## Running

```bash
streamlit run app.py
```

Your browser opens automatically (if not, open the
`http://localhost:8501` URL printed in the terminal).
Adjust parameters in the left sidebar and every tab's figures and computed
results update in real time.

---

## Usage

1. **Pick a mode** in the left sidebar (Dolan bridge / Manhattan).
2. **Set the resist and evaporation parameters.**
   - Common: PMMA thickness, MMA thickness, undercut, Evaporation 1 (θ₁ / φ₁ /
     metal thickness d₁)
   - Dolan: Evaporation 2 (θ₂ / φ₂ / d₂), bridge dimensions
   - Manhattan: Evaporation 2 (θ₂ / φ₂ / d₂), x/y arm opening widths
3. Choose the **Ray-scan resolution** (voxel density: accuracy vs. speed).
4. Inspect the cross-section, top view, and junction in the tabs. Use
   **Break Check** for the open/short verdict.
5. In the **Parameter Scan** tab, set the range, number of points, and
   resolution, then click **▶ Run scan**.
6. Export results as JSON in the **Junction Area** tab.

> Higher resolution (e.g. `Maximum (slowest)`) is slower. Start with
> `Standard (fast)` to get a feel, then refine as needed.
> A 2D scan runs the engine `points × points` times, so watch the point count.

---

## Key parameters

| Parameter | Description |
|---|---|
| `t_pmma` | PMMA (top resist) thickness [nm] |
| `t_mma` | MMA (bottom resist) thickness = air-gap height under the bridge [nm] |
| `undercut` | One-sided MMA undercut [nm] |
| θ₁ / φ₁ / d₁ | Polar angle / azimuth / metal thickness of evaporation 1 |
| θ₂ / φ₂ / d₂ | Polar angle / azimuth / metal thickness of evaporation 2 |
| `bridge_len` / `bridge_w` | (Dolan) bridge length / width [nm] |
| `manhattan_wx` / `manhattan_wy` | (Manhattan) x / y arm opening widths [nm] |

Beam direction: `beam = (sinθcosφ, sinθsinφ, −cosθ)` (θ measured from the normal).

---

## File overview

| File | Role |
|---|---|
| `app.py` | Streamlit UI (tabs, sidebar, scan, export) |
| `deposition3d.py` | **3D voxel evaporation engine** (`simulate` / `junction_footprint`) — source of truth |
| `voxel_view.py` | Cross-section / top-view rendering of the 3D result |
| `junction_area.py` | Analytic junction-area model (auxiliary / estimate) |
| `process_engine.py` | `ProcessParams` dataclass and geometry helpers |
| `cross_section.py` / `phi_cross_section.py` / `top_view.py` | 2D plotting utilities |
| `manhattan_check.py` | Manhattan open/short check |
| `gds_parser.py` / `generate_sample_gds.py` | GDS loading / sample generation (optional) |
| `requirements.txt` | Dependencies |

---

## Notes and assumptions

- Critical current uses `Ic[µA] = (junction area [nm²]) × 1e-4`
  (jc = 10 kA/cm², a rule of thumb for Al at ~4 K). It varies with material and
  conditions, so treat absolute values as estimates.
- A coarse voxel resolution introduces quantization error in the area and the
  open/short verdict. Near the boundary conditions, increase the resolution to
  confirm.
- The analytic model (`junction_area.py`) is for estimates and trend
  exploration; the final verdict is based on the 3D engine (`deposition3d.py`).
