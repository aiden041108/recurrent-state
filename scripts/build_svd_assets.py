"""Build publication plots from the supplied SVD experiment CSVs.

Temporal lines contain all 256 measured tokens (no smoothing or simplification).
Budget plots average the per-state errors over both heads and all tokens. The
control is selected separately for every state, within the same quantization
grouping and at or below the SVD storage cap, among the tested candidates.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'urp-svd-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    assets = args.repo / 'assets'
    assets.mkdir(parents=True, exist_ok=True)

    def read_csv(name):
        with (args.source / name).open(newline='') as handle:
            return list(csv.DictReader(handle))

    energy = read_csv('exp2_energy.csv')
    residual = read_csv('exp2_residual_metrics.csv')
    budget = read_csv('exp3_budget_comparison.csv')
    manifest = json.loads((args.source / 'svd_analysis_manifest.json').read_text())
    validation = json.loads((args.source / 'svd_analysis_validation.json').read_text())
    layers, ranks, tokens = [0, 8, 16], [1, 2, 4, 6], list(range(1, 257))
    assert manifest['state_shape'] == [256, 3, 2, 128, 128]
    assert len(energy) == len(residual) == 6144
    assert len(budget) == 24576
    assert all(row['zero_state'] == 'False' for rows in (energy, residual, budget) for row in rows)

    def select(rows, **conditions):
        return [row for row in rows if all(str(row[key]) == str(value) for key, value in conditions.items())]

    def values(rows, key):
        result = np.array([float(row[key]) for row in rows])
        assert np.isfinite(result).all()
        return result

    def series(rows, **conditions):
        selected = sorted(select(rows, **conditions), key=lambda row: int(row['tokens']))
        assert [int(row['tokens']) for row in selected] == tokens
        return selected

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 12,
        'axes.titlesize': 14, 'axes.titleweight': 'medium',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#cbd5e1', 'axes.labelcolor': '#344256',
        'xtick.color': '#526175', 'ytick.color': '#526175', 'text.color': '#172b4d',
        'grid.color': '#e5eaf0', 'svg.fonttype': 'none', 'svg.hashsalt': 'urp-svd-results',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'savefig.bbox': 'tight', 'savefig.pad_inches': .18,
        'path.simplify': False,
    })
    rank_colors = {1: '#243b63', 2: '#3b82c4', 4: '#148579', 6: '#8a5ca2'}
    layer_colors = {0: '#243b63', 8: '#3b82c4', 16: '#148579'}
    method_colors = {'best_direct_error': '#a4afbe', 'best_channel_error': '#3b82c4', 'svd_error': '#148579'}
    method_labels = {'best_direct_error': 'Best direct', 'best_channel_error': 'Best channel preservation', 'svd_error': 'SVD + residual'}
    summaries = []

    def save(fig, name):
        for extension in ('svg', 'png'):
            target = assets / f'{name}.{extension}'
            fig.savefig(target, dpi=160, metadata={'Date': None} if extension == 'svg' else None)
            if extension == 'svg':
                target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharey=True, layout='constrained')
    for layer, ax in zip(layers, axes):
        for rank in (1, 2, 4):
            rows = series(energy, layer=layer, head=0, rank=rank)
            data = values(rows, 'energy_retained')
            line, = ax.plot(tokens, data * 100, label=f'Rank {rank}', color=rank_colors[rank], lw=1.8)
            line.set_gid(f'layer-{layer}-head-0-rank-{rank}-all-256-tokens')
            summaries.append({'layer': layer, 'head': 0, 'rank': rank,
                              'final_energy': float(data[-1]), 'mean_energy': float(data.mean()),
                              'min_energy': float(data.min()), 'min_token': int(data.argmin() + 1),
                              'increases': int(np.count_nonzero(np.diff(data) > 1e-10)),
                              'decreases': int(np.count_nonzero(np.diff(data) < -1e-10))})
        ax.set(title=f'Layer {layer}', xlim=(1, 256), ylim=(0, 102), xticks=[1, 64, 128, 192, 256])
        ax.grid(axis='y', alpha=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Energy retained (%)')
    axes[0].legend(frameon=False, loc='lower left', fontsize=11)
    fig.supxlabel('Token position · Head 0 · all 256 states', fontsize=12)
    save(fig, 'svd-energy-trajectory')

    fig, ax = plt.subplots(figsize=(9.8, 3.9), layout='constrained')
    residual_summary = []
    for layer in layers:
        rows = series(residual, layer=layer, head=0, rank=4)
        data = values(rows, 'residual_norm_ratio')
        line, = ax.plot(tokens, data * 100, label=f'Layer {layer}', color=layer_colors[layer], lw=1.75)
        line.set_gid(f'layer-{layer}-head-0-rank-4-all-256-tokens')
        residual_summary.append({'layer': layer, 'head': 0, 'rank': 4,
                                 'final_residual_norm_ratio': float(data[-1]),
                                 'mean_residual_norm_ratio': float(data.mean()),
                                 'max_factor_rounding_relative_error': float(values(rows, 'factor_rounding_relative_error').max())})
    ax.set(xlabel='Token position · Head 0 · all 256 states', ylabel='Residual / original norm (%)',
           xlim=(1, 256), xticks=[1, 64, 128, 192, 256], ylim=(0, None))
    ax.grid(axis='y', alpha=.7)
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.set_axisbelow(True)
    save(fig, 'svd-residual-trajectory')

    aggregate = []
    for bits in (4, 8):
        for grouping in ('row', 'column'):
            for layer in layers:
                for rank in ranks:
                    rows = select(budget, bits=bits, grouping=grouping, layer=layer, rank=rank)
                    assert len(rows) == 512
                    assert {(int(row['tokens']), int(row['head'])) for row in rows} == {(t, h) for t in tokens for h in (0, 1)}
                    cap = {int(row['total_bytes']) for row in rows}
                    assert len(cap) == 1
                    assert all(int(row['best_direct_bytes']) <= int(row['total_bytes']) and int(row['best_channel_bytes']) <= int(row['total_bytes']) for row in rows)
                    item = {'bits': bits, 'grouping': grouping, 'layer': layer, 'rank': rank,
                            'state_count': len(rows), 'total_bytes': cap.pop(),
                            'effective_bits_per_entry': float(rows[0]['effective_bits_per_entry'])}
                    for key in ('svd_error', 'best_direct_error', 'best_channel_error', 'same_bits_direct_error'):
                        data = values(rows, key)
                        item[key] = float(data.mean())
                        item[f'{key}_median'] = float(np.median(data))
                    for baseline in ('direct', 'channel'):
                        errors = values(rows, f'best_{baseline}_error')
                        svd = values(rows, 'svd_error')
                        item[f'gain_vs_{baseline}_percent_from_mean_errors'] = float(100 * (1 - svd.mean() / errors.mean()))
                        item[f'wins_vs_{baseline}'] = int(np.count_nonzero(svd < errors))
                    aggregate.append(item)

    def group_summary(**conditions):
        found = select(aggregate, **conditions)
        assert len(found) == 1
        return found[0]

    for bits in (4, 8):
        primary_rank = 6 if bits == 4 else 4
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True, layout='constrained')
        positions = np.arange(3)
        methods = ['best_direct_error', 'best_channel_error', 'svd_error']
        largest = max(group_summary(bits=bits, grouping=g, layer=l, rank=primary_rank)[k] * 100 for g in ('row', 'column') for l in layers for k in methods)
        for grouping, ax in zip(('row', 'column'), axes):
            for i, method in enumerate(methods):
                heights = [group_summary(bits=bits, grouping=grouping, layer=layer, rank=primary_rank)[method] * 100 for layer in layers]
                bars = ax.bar(positions + (i - 1) * .25, heights, width=.23, color=method_colors[method], label=method_labels[method], zorder=3)
                ax.bar_label(bars, labels=[f'{v:.2f}' for v in heights], padding=4, fontsize=10.5)
            ax.set(title=f'{grouping.capitalize()}-wise quantization', xticks=positions,
                   xticklabels=[f'Layer {layer}' for layer in layers], ylim=(0, largest * 1.17))
            ax.grid(axis='y', alpha=.75)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Mean relative reconstruction error (%)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=11)
        cap = group_summary(bits=bits, grouping='row', layer=0, rank=primary_rank)['total_bytes']
        fig.suptitle(f'{bits}-bit residual + rank {primary_rank} · same storage cap: {cap:,} bytes', fontsize=15)
        save(fig, f'svd-budget-{bits}bit')

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex=True, layout='constrained')
    for gi, grouping in enumerate(('row', 'column')):
        for li, layer in enumerate(layers):
            ax = axes[gi, li]
            items = [group_summary(bits=4, grouping=grouping, layer=layer, rank=rank) for rank in ranks]
            for method in ('best_direct_error', 'best_channel_error', 'svd_error'):
                ax.plot([x['effective_bits_per_entry'] for x in items], [x[method] * 100 for x in items],
                        marker='o', color=method_colors[method], label=method_labels[method], lw=1.8, ms=4)
            ax.set(title=f'Layer {layer} · {grouping}-wise', xticks=[4.5, 4.75, 5.25, 5.75], ylim=(0, None))
            ax.tick_params(axis='x', labelsize=10)
            ax.grid(axis='y', alpha=.7)
            ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean relative error (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=11)
    fig.supxlabel('Storage cap (effective bits per entry) · ranks 1, 2, 4, 6', fontsize=12)
    save(fig, 'svd-budget-ranks')

    source_files = ['exp2_energy.csv', 'exp2_residual_metrics.csv', 'exp3_budget_comparison.csv',
                    'exp3_reconstruction.csv', 'svd_analysis_manifest.json', 'svd_analysis_validation.json',
                    'recurrent_state_trajectory.npz', 'svd_factors_fp16.npz']
    source_hashes = {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest() for name in source_files}
    result = {
        'schema_version': 1,
        'source_files_sha256': source_hashes,
        'source_state_sha256': manifest['source_state_sha256'],
        'state_shape': manifest['state_shape'],
        'validation': validation,
        'temporal_aggregation': 'Head 0, all 256 tokens. Lines join consecutive measured states; no smoothing.',
        'budget_aggregation': 'Arithmetic mean of per-state relative Frobenius errors over all 256 tokens and heads 0 and 1 (512 states per layer).',
        'budget_comparison': manifest['budget_comparison'],
        'budget_unit': manifest['budget'],
        'budget_control_scope': 'Same row/column quantization grouping. Candidate must fit at or below each SVD cap. Best tested candidate is selected separately per state.',
        'temporal_energy_summary': summaries,
        'rank4_residual_summary': residual_summary,
        'budget_means': aggregate,
        'limitations': manifest['limitations'],
    }
    (assets / 'svd-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'charts': ['svd-energy-trajectory', 'svd-residual-trajectory', 'svd-budget-4bit', 'svd-budget-8bit', 'svd-budget-ranks'],
                      'temporal_curves': 12, 'tokens_per_curve': 256, 'budget_aggregate_rows': len(aggregate),
                      'source_npz_sha256': source_hashes['recurrent_state_trajectory.npz']}))


if __name__ == '__main__':
    main()
