# Pre-Experiment URP

## Inside the Recurrent State

A concise experiment report on outliers, spectral concentration, and static quantization in pretrained Qwen3.5-0.8B.

**Website:** https://aiden041108.github.io/pre-experiment-urp/

The interactive Plotly surface shows actual FP32 128×128 recurrent states after every input token. Drag the slider to update; autoplay moves forward only and stops at token 256. Six layer/head pairs load separately (~16 MiB each), with at most two pairs retained in memory. Signed/magnitude and fixed per-pair/global ranges are available. Lower grid resolution affects the display only. The live camera and selected Zoom/Pan/Turntable tool persist through playback; Reset view restores only the initial camera.

## Data

- Model: Qwen/Qwen3.5-0.8B; pinned revision in `downloads/run_metadata.json`.
- Supplied run: Tesla T4, FP16 weights, native FP32 recurrent state.
- 256 tokens × 3 layers (0,8,16) × 2 heads (0,1) = 1,536 full matrices.
- Temporal energy, max/RMS, and singular-vector IPR plots use every token (1–256) for Head 0: 768 measured states across three layers.
- Quantization comparisons use the final state; original checkpoint diagnostics remain available separately.
- Recomputed CSV metrics and quantization results match the supplied files; checkpoint matrices match the full trajectory exactly. See `assets/numeric_audit.json`.
- Full NPZ: https://github.com/aiden041108/pre-experiment-urp/releases/download/v1.0.0/recurrent_state_trajectory.npz

The published values come from the new FP16 per-token run, not the earlier BF16 checkpoint run. No synthetic states or interpolated trajectories are used.

## Run locally

```sh
python3 -m http.server 8000
```

Then open http://localhost:8000/. A web server is required for binary asset loading; opening index.html as a file will not load the 3D data.

## Regenerate figures and browser assets

```sh
python3 -m pip install numpy matplotlib
python3 scripts/build_assets.py --data /path/to/qwen_state_trajectory_results
```

The build also recomputes all-token temporal metrics in `assets/temporal_metrics.json`. The temporal figures can be regenerated separately with `python3 scripts/build_temporal_assets.py`.

No frontend build is needed. GitHub Pages serves the main branch root (`.nojekyll`). The raw NPZ is a release asset; the repository contains split lossless FP32 files for the viewer.

## Scope and attribution

This is an observational, single-document preliminary experiment. It does not reproduce LeapQuant, establish the origin of outliers, or evaluate downstream model quality. Static quantization does not feed errors into recurrent updates. Original min–max grids may not represent zero exactly; an independent zero-point diagnostic is included separately.

Visual direction inspired by https://leapquant.github.io/; the page and figures are original. Plotly.js is distributed under its MIT license (see `vendor/LICENSE-plotly.txt`). Fonts are served by Google Fonts. Model and implementation references are linked on the website.

## Verification

Local asset links, English page text, data shape and CSV values checked. Browser checks covered immediate slider updates, signed/magnitude, layer/head changes, fixed scale, forward-only stop at 256, and mobile overflow. JavaScript data/control tests also used the actual FP32 files. See the supplied validation report for model-run consistency checks and numerical tolerances.
