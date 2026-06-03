# ShadowCast — Josephson Junction Shadow-Evaporation Simulator

A Streamlit app that simulates the fabrication of **Josephson junctions** by
**shadow / oblique evaporation** in 3D. It supports both the **Dolan bridge**
and **Manhattan** geometries, and computes — from the resist profile and
evaporation angles — the junction overlap area, critical current, inductance,
and Josephson energy.

The physics core is a 3D voxel ray-cast engine (`deposition3d.py`): it
ray-traces the tilted evaporation beams into a voxel grid to find where each
metal film is deposited, then extracts the junction as the **full 3D oxide
barrier** between the two electrodes — both on the substrate floor *and* on the
vertical metal sidewalls of the lower electrode. This 3D engine is the
**source of truth** — all on-screen views and judgments are based on it.

---

## Features

- **Two fabrication modes**
  - **Dolan bridge**: uniaxial tilt (same φ, opposite θ) so the deposition wraps
    under the suspended bridge.
  - **Manhattan**: two evaporation beams (with independently configurable
    θ₁/φ₁ and θ₂/φ₂) that cross each other.
- **Bilayer or trilayer deposition stack** (works in either mode)
  - **Bilayer**: evap 1 → oxidation → evap 2 (the classic two-electrode setup).
  - **Trilayer**: Nb→Al → oxidation → Al→Nb (Nb/Al/Al/Nb). The oxide forms on
    the exposed Nb *and* Al of the bottom electrode, and the junction barrier is
    classified by the metal pair across the oxide — **Nb-Al / Al-Al / Nb-Nb** —
    each reported and coloured separately. The same metal deposited in different
    evaporations is distinguished by **brightness** in the views. Per-sublayer
    thicknesses and the evap-2 / evap-4 tilt angles are configurable (defaulting
    to their electrode's primary angle).
- **3D shadow-evaporation engine**: voxel ray-casting that reproduces the metal
  films, the oxide, and the resist undercut.
- **Arbitrary junction shape, full 3D barrier**: the junction is not assumed to
  be square — it is the true oxide barrier between the two electrodes (even when
  non-rectangular), measured by cell count in 3D. The barrier counts **both the
  substrate floor and the vertical metal sidewalls**, since the film walls also
  form junctions. A device may contain several spatially separate junctions
  (e.g. one on each side of a Dolan bridge); each is reported individually.
- **Electrical quantities** derived from the junction area:
  - Critical current Ic (Ambegaokar–Baratoff, jc = 10 kA/cm²)
  - Josephson inductance L_J = ħ / (2e·Ic)
  - Josephson energy E_J = (Φ₀/2π)·Ic (also shown as E_J/h [GHz] and E_J/k_B [K])
- **Seven visualization tabs**
  1. **📐 Cross-section** — cross-section that can be rotated to any in-plane
     slice angle α (**signed, −90 … 90°, default 0**: 0 = x–z, ±90 = y–z) and
     offset (with evaporation-beam arrows)
  2. **🗺️ Top View** — top view (metal films, shadow, undercut, junction region),
     plus the **lift-off film-thickness map**: a 2D heat map and a 3D surface
     where the *z value is the stacked metal thickness* (electrode overlap reads
     thicker)
  3. **🔄 φ Junction View** — zoomed top view around the junction
  4. **🔍 Break Check** — open/short verdict and electrical metrics
  5. **📈 Parameter Scan** — parameter sweep (see below)
  6. **🌐 Wafer Map** — JJ-area variation across the wafer for a
     fixed-source / tilted-wafer (Plassys) evaporator, drawn on a **real wafer
     disk with its primary flat (オリフラ)**, with the per-position **effective
     deposition angles and junction area** exposed via a hover map and a
     sortable table (see below)
  7. **📊 Junction Area** — full parameter summary and result export
- **Parameter Scan**
  - **1D / 2D** sweeps (2D rendered as a heatmap)
  - The **value range and number of points** are configurable per variable
  - The sweep **voxel density (resolution)** offers the same 5 presets as the
    sidebar
  - The output plots **junction area, Ic, L_J, E_J/h, and E_J/k_B all stacked
    vertically**
  - For Manhattan, the per-beam **θ₁ / φ₁ / θ₂ / φ₂** are also sweepable
- **Wafer Map (Plassys point-source / tilted-wafer)**
  - Models a real oblique evaporator: a **fixed point source** with the **wafer
    tilted** to hit the nominal (θ, φ) *at the wafer centre*. Because the source
    is at a finite throw distance, an off-centre device sees a slightly different
    local angle, so the **junction area drifts with wafer position**.
  - Draws an **actual wafer disk** (2 / 3 / 4 / 6-inch, default **4-inch**) with
    its **primary flat (オリフラ)** at the bottom, replicates the *same device*
    over an **N×N grid** clipped to the wafer (off-wafer cells are skipped), and
    plots a 2D heat map of **junction area** and **critical current Ic**, plus
    spread statistics (min / max / mean, `(max−min)/mean`, std/mean).
  - Surfaces the **per-position effective deposition angles (θ′, φ′) for every
    evaporation** alongside the area/Ic: hover any cell of an interactive map for
    its values, or read the full **sortable table** (also CSV-exportable).
  - Draws a **per-evaporation schematic** of the fixed source and how the wafer
    is tilted for each evaporation.
  - The single-JJ simulation is **unchanged**; the wafer centre reproduces it.
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
6. In the **Wafer Map** tab, set the throw distance L, wafer size (2/3/4/6-inch),
   grid size N, and resolution, then click **▶ Run wafer map** to see how the
   junction area varies across the wafer; hover a cell or read the table for the
   per-position effective angles.
7. Export results as JSON in the **Junction Area** tab.

> Higher resolution (e.g. `Maximum (slowest)`) is slower. Start with
> `Standard (fast)` to get a feel, then refine as needed.
> A 2D scan runs the engine `points × points` times, and a wafer map runs it
> `N × N` times, so watch the point / grid count.

---

## Wafer Map (Plassys point-source model)

In a real oblique evaporator the metal comes from a **fixed point source**, and
the **wafer is tilted/rotated** so the beam meets the wafer normal at the nominal
(θ, φ) **at the wafer centre**. Because the source is at a finite throw distance
`L`, a device patterned off-centre on the wafer sees the beam arrive at a
slightly different **local** angle (θ′, φ′), which shifts each electrode's shadow
offset — so the junction area drifts with wafer position.

Geometry (lab frame): the source sits at the origin and the wafer centre is held
at `(0, 0, −L)`; for each evaporation the wafer is tilted by `R = Ry(θ)·Rz(−φ)`
(a fixed tilt axis plus a wafer spin that sets the azimuth — the Plassys stage),
chosen so the vertical beam reproduces `beam_direction(θ, φ)` at the centre. A
device at wafer position `(X, Y)` (in mm) maps to the lab position
`P = C + X·eX + Y·eY`; the local beam `(P − S)/|P − S|`, expressed back in the
wafer frame, gives the localized (θ′, φ′) fed to the engine. The deviation
scales roughly as `r / L` (wafer radius over throw distance), and at the centre
the result is identical to the single-JJ simulation.

Each grid cell runs the full 3D engine with its localized angles, so the map
works for every mode and stack with no change to the physics. Only the beam
angles vary with position — the nm-scale device/resist geometry is unchanged.

The grid is drawn on an **actual wafer disk** (2 / 3 / 4 / 6-inch, default
4-inch) with its **primary flat** at the bottom (the chord sits at `y = −d`,
`d = √(R² − (chord/2)²)`); cells outside the disk or below the flat are skipped.
Besides the area/Ic heat maps, the **localized (θ′, φ′) of every evaporation** is
reported per position — hover any cell of the interactive map, or read/sort the
full table (CSV-exportable). The wafer centre always reproduces the single-JJ
result.

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
| `stack` | `Bilayer` or `Trilayer` (Nb/Al/Al/Nb) |
| `tri_t1…tri_t4` | (Trilayer) Nb/Al/Al/Nb sublayer thicknesses [nm] |
| `tri_angle2` / `tri_angle4` | (Trilayer) evap-2 (Al) / evap-4 (Nb) tilt angles [°] |

Beam direction: `beam = (sinθcosφ, sinθsinφ, −cosθ)` (θ measured from the normal).

---

## File overview

| File | Role |
|---|---|
| `app.py` | Streamlit UI (tabs, sidebar, scan, export) |
| `deposition3d.py` | **3D voxel evaporation engine** (`simulate` / `junction_footprint`) — source of truth |
| `voxel_view.py` | Cross-section / top-view rendering of the 3D result, plus the lift-off thickness map (2D + 3D) and the per-evaporation source / wafer-tilt schematic |
| `junction_area.py` | Analytic junction-area model (auxiliary / estimate) |
| `process_engine.py` | `ProcessParams` dataclass, shadow geometry, and the Plassys wafer-position helpers (`wafer_local_angles` / `wafer_params`) |
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
- The **Wafer Map** assumes an ideal point source, the wafer tilted about its own
  centre, and beam angles that vary purely with wafer position (the resist/device
  geometry is identical at every position). At short throw distances or large
  off-centre offsets the local angle can shift enough to collapse the junction
  to zero area — a real effect, not a bug.
