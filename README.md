# ShadowCast — Josephson Junction Shadow-Evaporation Simulator

A Streamlit app that simulates the fabrication of **Josephson junctions** by
**shadow / oblique evaporation** in 3D. It supports both the **Dolan bridge**
and **Manhattan** geometries, and computes — from the resist profile and
evaporation angles — the junction overlap area, critical current, normal-state
resistance, inductance, and Josephson energy.

The physics core is a 3D voxel ray-cast engine (`deposition3d.py`): it
ray-traces the tilted evaporation beams into a voxel grid to find where each
metal film is deposited, then extracts the junction as the **oxide barrier**
between the two electrodes. By default the junction area is the **horizontal
floor overlap** (electrode 2 sitting on top of electrode 1 across the oxide),
with an **opt-in to also count the vertical metal sidewalls** of the lower
electrode. This 3D engine is the **source of truth** — all on-screen views and
judgments are based on it.

---

## Features

- **Two fabrication modes**
  - **Dolan bridge**: uniaxial tilt (same φ, opposite θ) so the deposition wraps
    under the suspended bridge.
  - **Manhattan**: two evaporation beams (with independently configurable
    θ₁/φ₁ and θ₂/φ₂) that cross each other. The MMA slider sets the lower
    undercut sublayer and the PMMA slider the upper imaging resist.
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
- **Rounded resist corners** (option): a slider rounds the **top resist layer's
  top and bottom faces** with a fillet of the chosen radius (0 = sharp). The
  fillet is a true circle defined **independently of the voxel grid density**
  (the shadow test is analytic), so the modelled shape — and the resulting
  shadow / junction area — does not change with resolution; the cross-section
  fill is still drawn at voxel resolution. Reflected consistently in the
  cross-section, top view, and wafer map.
- **Arbitrary junction shape**: the junction is not assumed to be square — it is
  the true oxide barrier between the two electrodes (even when non-rectangular),
  measured by cell count in 3D. A device may contain several spatially separate
  junctions (e.g. one on each side of a Dolan bridge); each is reported
  individually.
- **Floor-only vs floor + sidewall area** (toggle): by default the area is the
  **horizontal floor overlap** only (the expected planar junction). Enable
  **“Count sidewall (vertical) junction area”** to also include the vertical
  M-O-M barrier where metal climbs the resist sidewall; the floor / sidewall /
  total areas are then shown separately and the total drives the electrical
  quantities.
- **Soft-edge (finite-source penumbra) deposition** (opt-in): models the **real
  Plassys source** — its **e-beam raster pattern** (rotating line / uniform disk /
  Gaussian, spot ≈10–15 mm) at the **throw distance** (≈550 mm) — by integrating
  occlusion over the resulting beam-direction cloud (the same exact geometry the
  finite-source Monte-Carlo uses). The deposited film **thickness tapers near the
  shadow edge** (penumbra ≈ source size / L), reproducing the **rounded (tapered)
  metal edge**, especially with a rounded resist lip. The junction footprint/area
  is set by the central beam (so the area is preserved). Visible at finer
  resolution (the film must be several voxels thick); slower (multi-ray).
- **Side-wall deposition effect** (opt-in): the first evaporation also coats the
  resist sidewall, narrowing the opening seen by later evaporations. The
  narrowing grows with the local incident angle, so it makes the across-wafer
  area / R_n map **asymmetric** (after *Jpn. J. Appl. Phys.* 10.35848/1347-4065/aca256).
- **Electrical quantities** derived from the junction area:
  - Critical current Ic = Jc × area, with a **configurable critical-current
    density Jc** (default **1 kA/cm²**, typical Al-AlOx-Al range 100–10 000 A/cm²)
  - Normal-state resistance R_n = πΔ/(2e·Ic) (**Ambegaokar–Baratoff**, Al gap
    Δ ≈ 0.18 meV), shown in kΩ
  - Josephson inductance L_J = ħ / (2e·Ic)
  - Josephson energy E_J = (Φ₀/2π)·Ic (also shown as E_J/h [GHz] and E_J/k_B [K])
- **Finite e-beam source Monte-Carlo** (opt-in): model the e-beam source with a
  finite spatial spread set by its raster pattern instead of an ideal point —
  **point / rotating line (disk, areal density ∝ 1/ρ, default) / uniform disk /
  Gaussian** — at a configurable throw distance. Each Monte-Carlo draw jitters
  every evaporation's beam angle and re-runs the engine, building the JJ-area
  distribution (mean ± σ, with a histogram). Available for the single junction
  (🎲 Source MC tab) and per cell on the Wafer Map.
- **Beam-angle distribution & correlation viewer**: from the Monte-Carlo draws,
  inspect the θ₁ / φ₁ / θ₂ / φ₂ distributions as 1-D histograms, 2-D scatter of
  any pair, or a correlation matrix (per-cell on the Wafer Map).
- **Nine visualization tabs**
  1. **📐 Cross-section** — cross-section that can be rotated to any in-plane
     slice angle α (**signed, −90 … 90°, default 0**: 0 = x–z, ±90 = y–z) and
     offset (with evaporation-beam arrows). The lift-off panel colours the
     thin **metal–oxide–metal band** on both oxide sides of the junction, and a
     **process-stages figure** (resist → evap 1 → oxidation → evap 2 → lift-off)
     can be panned/zoomed.
  2. **🗺️ Top View** — top view (metal films, shadow, undercut, junction region),
     plus the **lift-off film-thickness map**: a 2D heat map and a **drag-rotatable
     3D surface (Plotly)** where the *z value is the stacked metal thickness*
     (electrode overlap reads thicker). Both panels share the same field and show
     only the deposited film (the zero baseline is masked out).
  3. **🎬 Playback** — step-through of the deposition: scrub a frame slider to
     watch each evaporation's film **grow layer-by-layer** toward the source, then
     oxidation and lift-off (reuses the Cross-section slice/view; optional GIF).
  4. **🔄 φ Junction View** — zoomed top view around the junction
  5. **🔍 Break Check** — open/short verdict and electrical metrics
  6. **📈 Parameter Scan** — parameter sweep (see below)
  7. **🎲 Source MC** — finite e-beam source Monte-Carlo for the single junction
     (beam-pattern controls, mean ± σ, area histogram, beam-angle viewer)
  8. **🌐 Wafer Map** — JJ-area (and more) variation across the wafer for a
     fixed-source / tilted-wafer (Plassys) evaporator, drawn on a **real wafer
     disk with its primary flat (オリフラ)** (see below)
  9. **📊 Junction Area** — full parameter summary and result export
- **Parameter Scan**
  - **1D / 2D** sweeps (2D rendered as a heatmap)
  - The **value range and number of points** are configurable per variable
  - The sweep **voxel density (resolution)** offers the same presets as the
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
    over a **centred N×N grid** clipped to the wafer (off-wafer cells skipped).
    The **grid cell pitch** (1–10 mm) is adjustable and the **max grid N scales
    with the cell size** so the grid can be spread to fill the whole wafer.
  - **Selectable 2D map quantity** — colour the interactive heatmap by JJ area,
    Est. Ic, **R_n [kΩ]**, **Overlap x / y**, the per-evaporation **effective
    angles θ′ / φ′**, the per-evaporation **source distance**, or (with the
    finite source) the area σ. A **colour-map picker** (default
    green → yellow → red, low → high) and an option to **print the value inside
    each cell** (selectable significant figures) are provided; the wafer renders
    as a true circle.
  - Reports spread statistics (min / max / mean, `(max−min)/mean`, std/mean) and
    a **sortable, CSV-exportable table** of every on-wafer cell (area, Ic, R_n,
    per-evap θ′/φ′ and source distance).
  - Optional **finite-source Monte-Carlo per cell** (beam-pattern source) gives a
    per-cell area σ map and a per-cell beam-angle viewer.
  - Draws a **per-evaporation schematic** of the fixed source and how the wafer
    is tilted for each evaporation.
  - The single-JJ simulation is **unchanged**; the wafer centre reproduces it.
- **Save / load parameters**: store and restore settings as JSON, plus a
  reset-to-defaults button
- **GDS import** (optional): reads GDSII layout files (requires `gdstk`)

---

## Requirements

- **Python 3.10+** (developed and tested on 3.11)
- Core packages: `streamlit`, `numpy`, `matplotlib`, `plotly` (plus `pandas` and
  `altair`, which ship with Streamlit and back the interactive wafer-map
  heatmap/table)
- Optional: `gdstk` for GDS import

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
plotly>=5.0.0         # drag-rotatable 3D thickness surface (Top View tab)
gdstk>=0.9.0          # only if you use GDS import
```

(`pandas` and `altair` come bundled with Streamlit.)

---

## Running

```bash
streamlit run app.py
```

Your browser opens automatically (if not, open the
`http://localhost:8501` URL printed in the terminal).
Set the parameters in the left sidebar and press **▶ Run simulation** to run the
3-D engine. Geometry / angle / resolution changes take effect only on the next
Run (a notice appears when inputs have changed), so dragging a slider never
re-simulates. Display / measurement controls — critical-current density `Jc`,
the sidewall-area toggle, the cross-section view angle/zoom, and the wafer-map
colour/metric — update **live** without re-running the engine.

---

## Usage

1. **Pick a mode** in the left sidebar (Dolan bridge / Manhattan).
2. **Set the resist and evaporation parameters.**
   - Common: PMMA thickness, MMA thickness, undercut, Evaporation 1 (θ₁ / φ₁ /
     metal thickness d₁)
   - Dolan: Evaporation 2 (θ₂ / φ₂ / d₂), bridge dimensions
   - Manhattan: Evaporation 2 (θ₂ / φ₂ / d₂), x/y arm opening widths
3. In **⚙ Process options**, optionally enable the **side-wall effect**, set the
   **critical-current density Jc**, and choose whether to **count the sidewall
   (vertical) junction area**.
4. Choose the **Ray-scan resolution** (voxel density: accuracy vs. speed).
5. Inspect the cross-section, top view, and junction in the tabs. Use
   **Break Check** for the open/short verdict.
6. (Optional) Enable the **finite e-beam source** and press **▶ Run source
   Monte-Carlo** to see the single-JJ area distribution in the **Source MC** tab.
7. In the **Parameter Scan** tab, set the range, number of points, and
   resolution, then click **▶ Run scan**.
8. In the **Wafer Map** tab, set the throw distance L, wafer size, grid N and
   cell pitch, and resolution, then click **▶ Run wafer map**. Choose the 2D-map
   quantity, colour map, and (optionally) in-cell value labels; hover a cell or
   read the table for the per-position effective angles, distances, and R_n.
9. Export results as JSON in the **Junction Area** tab.

> Higher resolution (e.g. `Maximum (slowest)`) is slower. Start with
> `Standard (fast)` to get a feel, then refine as needed.
> A 2D scan runs the engine `points × points` times, a wafer map runs it
> `N × N` times, and the finite-source Monte-Carlo multiplies that by `N_mc`, so
> watch the point / grid / sample count.

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
wafer frame, gives the localized (θ′, φ′) fed to the engine, and `|P − S|` gives
the per-position **source distance**. The deviation scales roughly as `r / L`
(wafer radius over throw distance), and at the centre the result is identical to
the single-JJ simulation.

Each grid cell runs the full 3D engine with its localized angles, so the map
works for every mode and stack with no change to the physics. Only the beam
angles vary with position — the nm-scale device/resist geometry is unchanged
(unless the side-wall effect is enabled, which couples the local angle to the
opening width).

The grid is **centred** on the wafer with an adjustable **cell pitch**; the
maximum grid N grows with the cell size so the grid can be spread to fill the
disk. The primary flat sits at `y = −d`, `d = √(R² − (chord/2)²)`; cells outside
the disk or below the flat are skipped. Besides JJ area, the heatmap can show
Ic, **R_n [kΩ]**, Overlap x/y, the localized (θ′, φ′) and source distance of
every evaporation, with a selectable colour map and optional in-cell value
labels. The wafer centre always reproduces the single-JJ result.

---

## Key parameters

| Parameter | Description |
|---|---|
| `t_pmma` | PMMA thickness [nm] (Dolan: top resist; Manhattan: upper imaging resist) |
| `t_mma` | MMA thickness [nm] (Dolan: bottom resist = bridge air-gap height; Manhattan: lower undercut sublayer) |
| `undercut` | One-sided MMA undercut [nm] |
| `resist_round` | Resist opening corner fillet radius [nm] (0 = sharp lip/foot) |
| `soft_edge` | Soft-edge (finite-source penumbra) deposition on/off |
| `soft_pattern` / `soft_size` / `soft_L` | Plassys source: e-beam pattern · spot size [mm] · throw distance [mm] (penumbra ≈ size/L) |
| θ₁ / φ₁ / d₁ | Polar angle / azimuth / metal thickness of evaporation 1 |
| θ₂ / φ₂ / d₂ | Polar angle / azimuth / metal thickness of evaporation 2 |
| `bridge_len` / `bridge_w` | (Dolan) bridge length / width [nm] |
| `manhattan_wx` / `manhattan_wy` | (Manhattan) x / y arm opening widths [nm] |
| `stack` | `Bilayer` or `Trilayer` (Nb/Al/Al/Nb) |
| `tri_t1…tri_t4` | (Trilayer) Nb/Al/Al/Nb sublayer thicknesses [nm] |
| `tri_angle2` / `tri_angle4` | (Trilayer) evap-2 (Al) / evap-4 (Nb) tilt angles [°] |
| `sidewall` | Enable the side-wall deposition effect (prior evap narrows later openings) |
| Jc | Critical-current density [A/cm²] used for Ic = Jc × area (default 1 kA/cm²) |
| Count sidewall area | Include the vertical sidewall barrier in the junction area (default off) |

Beam direction: `beam = (sinθcosφ, sinθsinφ, −cosθ)` (θ measured from the normal).

---

## File overview

| File | Role |
|---|---|
| `app.py` | Streamlit UI (tabs, sidebar, scan, finite-source MC, wafer map, export) |
| `deposition3d.py` | **3D voxel evaporation engine** (`simulate` / `junction_footprint` / `junction_combos`) — source of truth; floor-only vs sidewall measurement, side-wall occlusion |
| `voxel_view.py` | Cross-section / top-view rendering of the 3D result, the lift-off thickness map (2D + Plotly 3D), the wafer disk map, and the per-evaporation source / wafer-tilt schematic |
| `junction_area.py` | Analytic junction-area model (auxiliary / estimate) |
| `process_engine.py` | `ProcessParams` dataclass, shadow geometry, the Plassys wafer helpers (`wafer_local_angles` / `wafer_params` / `wafer_source_dist`), and the finite-source beam-pattern sampler (`sample_beam_cloud` / `wafer_params_source`) |
| `cross_section.py` / `phi_cross_section.py` / `top_view.py` | 2D plotting utilities |
| `manhattan_check.py` | Manhattan open/short check |
| `gds_parser.py` / `generate_sample_gds.py` | GDS loading / sample generation (optional) |
| `requirements.txt` | Dependencies |

---

## Notes and assumptions

- Critical current uses `Ic = Jc × (junction area)`, with **Jc configurable**
  (default 1 kA/cm², a typical Al-AlOx-Al value at ~4 K). It varies with material
  and oxidation, so treat absolute Ic / R_n values as estimates; tune Jc to match
  your measured R_n.
- By default the junction area is the **horizontal floor overlap** only; enabling
  “Count sidewall (vertical) junction area” adds the vertical M-O-M barrier and
  reports the floor / sidewall / total split (the total then drives Ic / R_n).
- A coarse voxel resolution introduces quantization error in the area and the
  open/short verdict. Near the boundary conditions, increase the resolution to
  confirm.
- The analytic model (`junction_area.py`) is for estimates and trend
  exploration; the final verdict is based on the 3D engine (`deposition3d.py`).
- The **Wafer Map** assumes an ideal point source (with an optional finite
  beam-pattern spread for the Monte-Carlo), the wafer tilted about its own
  centre, and beam angles that vary with wafer position. At short throw distances
  or large off-centre offsets the local angle can shift enough to collapse the
  junction to zero area — a real effect, not a bug.
