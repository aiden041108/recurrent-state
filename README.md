# Inside the Recurrent State

Pre-Experiment URP

A concise experiment report on outliers, spectral concentration, and static quantization in pretrained Qwen3.5-0.8B.

**Website:** https://aiden041108.github.io/recurrent-state/

The interactive Plotly surface shows actual FP32 128×128 recurrent states after every input token. Drag the slider to update; autoplay moves forward only and stops at token 256. Six layer/head pairs load separately (~16 MiB each), with at most two pairs retained in memory. Signed/magnitude and fixed per-pair/global ranges are available. Lower grid resolution affects the display only. The live camera and selected Zoom/Pan/Turntable tool persist through playback; Reset view restores only the initial camera.

## Data

- Model: Qwen/Qwen3.5-0.8B; pinned revision in `downloads/run_metadata.json`.
- Supplied run: Tesla T4, FP16 weights, native FP32 recurrent state.
- 256 tokens × 3 layers (0,8,16) × 2 heads (0,1) = 1,536 full matrices.
- Experiment 2 shows E1/E2/E4 for every token (1–256), Head 0.
- Experiment 3 analyzes all 1,536 matrices. Each layer summary averages 256 tokens × two heads. Exact per-state SVD factors are stored in FP16; residuals use symmetric zero-exact per-channel quantization.
- Direct and channel-preservation baselines are the best tested candidates within each SVD storage cap, selected per state with the same row/column grouping. Counts include theoretical packed codes, scales, factors, and indices.
- Main settings: 4-bit residual + rank 6, and 8-bit residual + rank 4; ranks 1/2/4/6 are all evaluated.
- Recomputed CSV metrics and quantization results match the supplied files; checkpoint matrices match the full trajectory exactly. See `assets/numeric_audit.json`.
- Full NPZ: https://github.com/aiden041108/recurrent-state/releases/download/v1.0.0/recurrent_state_trajectory.npz

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

Regenerate the current SVD figures with `python3 scripts/build_svd_assets.py --source /path/to/qwen_svd_exp23_results` (NumPy and Matplotlib). The supplied run reused the original recorded trajectory; it did not rerun model collection.

No frontend build is needed. GitHub Pages serves the main branch root (`.nojekyll`). The raw NPZ is a release asset; the repository contains split lossless FP32 files for the viewer.

## Scope and attribution

This is an observational, single-document preliminary experiment. It does not reproduce LeapQuant, establish the origin of outliers, or evaluate downstream model quality. Static quantization does not feed errors into recurrent updates. The current SVD experiment uses a zero-exact symmetric grid. Stored FP16 factors are used when computing the residual. Exact SVD and theoretical byte accounting do not measure runtime throughput or packed GPU memory. Channel controls retain a dense residual; stronger optimized codecs remain to be tested.

Visual direction inspired by https://leapquant.github.io/; the page and figures are original. Plotly.js is distributed under its MIT license (see `vendor/LICENSE-plotly.txt`). Fonts are served by Google Fonts. Model and implementation references are linked on the website.

## Verification

Local asset links, English page text, data shape and CSV values checked. Browser checks covered immediate slider updates, signed/magnitude, layer/head changes, fixed scale, forward-only stop at 256, and mobile overflow. JavaScript data/control tests also used the actual FP32 files. See the supplied validation report for model-run consistency checks and numerical tolerances.
