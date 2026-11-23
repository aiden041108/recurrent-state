"""Recompute every plotted token from the published FP32 recurrent states.

The temporal plots use head 0 at layers 0, 8, and 16. Each polyline contains
all 256 measurements; no smoothing, interpolation, or path simplification.
Original checkpoint diagnostics and final-state quantization remain intact.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'urp-full-token-matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--report', type=Path, help='Optional local validation report')
args = parser.parse_args()
assets = args.repo / 'assets'
metadata = json.loads((assets / 'experiment.json').read_text())
expected_shape = tuple(metadata['state_shape'])
assert expected_shape == (256, 3, 2, 128, 128)
tokens = metadata['token_indices']
assert tokens == list(range(1, 257))
layers = [0, 8, 16]
keys = ('E8', 'max_over_rms', 'u1_ipr', 'v1_ipr')
rows, hashes, summaries, checks = [], {}, [], []
for layer in layers:
    name = metadata['state_files'][f'L{layer}-H0']
    source = assets / name
    assert source.stat().st_size == 256 * 128 * 128 * 4
    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    states = np.memmap(source, dtype='<f4', mode='r', shape=(256, 128, 128))
    layer_rows = []
    for token, state in zip(tokens, states):
        x = np.asarray(state, dtype=np.float64)
        assert np.isfinite(x).all()
        u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
        energy = singular_values ** 2
        assert energy.sum() > 0, (layer, token, 'zero state has undefined relative diagnostics')
        row = {
            'tokens': token, 'layer': layer, 'head': 0,
            'E8': float(energy[:8].sum() / energy.sum()),
            'max_over_rms': float(np.abs(x).max() / np.sqrt(np.mean(x * x))),
            'u1_ipr': float(np.sum(u[:, 0] ** 4)),
            'v1_ipr': float(np.sum(vh[0] ** 4)),
        }
        assert all(np.isfinite(row[key]) for key in keys)
        rows.append(row)
        layer_rows.append(row)
    for old in metadata['metrics']:
        if old['layer'] != layer or old['head'] != 0:
            continue
        actual = layer_rows[old['tokens'] - 1]
        for key in keys:
            np.testing.assert_allclose(actual[key], old[key], rtol=1e-8, atol=1e-9)
        checks.append({'layer': layer, 'head': 0, 'tokens': old['tokens'], 'matches_original': True})
    summary = {'layer': layer, 'head': 0}
    for key in keys:
        values = np.array([row[key] for row in layer_rows])
        summary[key] = {
            'first': float(values[0]), 'last': float(values[-1]),
            'min': float(values.min()), 'min_token': int(values.argmin() + 1),
            'max': float(values.max()), 'max_token': int(values.argmax() + 1),
            'increases': int(np.count_nonzero(np.diff(values) > 1e-10)),
            'decreases': int(np.count_nonzero(np.diff(values) < -1e-10)),
        }
    summaries.append(summary)

assert len(rows) == 768 and len(checks) == 12
payload = {
    'schema_version': 1,
    'description': 'Actual per-token temporal diagnostics: all 256 saved states for head 0 in layers 0, 8, and 16.',
    'tokens': tokens, 'layer_ids': layers, 'head_ids': [0], 'analysis_snapshots': len(rows),
    'state_dtype': 'float32', 'calculation_dtype': 'float64',
    'formulas': {
        'E8': 'sum(singular_values[:8] ** 2) / sum(singular_values ** 2)',
        'max_over_rms': 'max(abs(S)) / sqrt(mean(S ** 2))',
        'u1_ipr': 'sum(U[:, 0] ** 4), where U has unit-norm columns',
        'v1_ipr': 'sum(Vh[0, :] ** 4), where Vh has unit-norm rows',
    },
    'source_sha256': hashes,
    'checkpoint_validation': checks,
    'metrics': rows,
}
(assets / 'temporal_metrics.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'axes.titleweight': 'medium',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#cbd5e1', 'axes.labelcolor': '#344256',
    'xtick.color': '#526175', 'ytick.color': '#526175', 'text.color': '#172b4d',
    'grid.color': '#e5eaf0', 'svg.fonttype': 'none', 'svg.hashsalt': 'urp-every-token',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.pad_inches': .15,
    'path.simplify': False,
})
colors = {0: '#243b63', 8: '#3b82c4', 16: '#148579'}
plots = [
    ('E8', 'energy', 'Spectral concentration', 'Top-8 energy retained (%)', 100),
    ('max_over_rms', 'magnitude', 'Relative entry magnitude', 'max |S| / RMS(S)', 1),
    ('u1_ipr', 'left-ipr', 'Key-direction localization', 'IPR(u₁)', 1),
    ('v1_ipr', 'right-ipr', 'Value-direction localization', 'IPR(v₁)', 1),
]
for key, name, title, ylabel, factor in plots:
    fig, ax = plt.subplots(figsize=(7.5, 4.15), layout='constrained')
    for layer in layers:
        series = [row for row in rows if row['layer'] == layer]
        line, = ax.plot(tokens, [row[key] * factor for row in series],
                        color=colors[layer], label=f'Layer {layer}', lw=1.65)
        line.set_gid(f'layer-{layer}-all-256-tokens')
        assert len(line.get_xdata()) == 256
    ax.set(xlabel='Tokens processed · every token', ylabel=ylabel,
           xticks=[1, 64, 128, 192, 256], xlim=(1, 256), title=title)
    ax.grid(axis='y', alpha=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='best')
    if key.endswith('ipr'):
        ax.axhline(1 / 128, color='#b8c1cb', lw=1, ls='--')
    for extension in ('svg', 'png'):
        target = assets / f'temporal-{name}.{extension}'
        fig.savefig(target, dpi=180,
                    metadata={'Date': None} if extension == 'svg' else None)
        if extension == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)

report = {'measurement_count': len(rows), 'points_per_layer_per_chart': 256,
          'all_12_previous_checkpoints_match': True, 'zero_states': 0,
          'summaries': summaries}
if args.report:
    args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
