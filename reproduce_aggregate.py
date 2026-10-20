from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data/aggregate')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    specs = {'segmentation_summary': (15, ['architecture', 'structure'], ['mean_dice', 'fold_sd', 'folds']), 'task_performance': (21, ['pathway', 'stage', 'endpoint'], ['value']), 'calibration_summary': (4, ['endpoint'], ['brier', 'slope']), 'knosp_error_summary': (3, ['outcome'], ['cases', 'percentage'])}
    tables, checks = ({}, {})
    for name, (count, keys, numbers) in specs.items():
        path = args.data_dir / f'{name}.csv'
        rows = read_rows(path)
        assert len(rows) == count, f'{name}: expected {count} rows'
        assert len({tuple((row[k] for k in keys)) for row in rows}) == count, f'{name}: duplicate keys'
        for row in rows:
            assert all((math.isfinite(float(row[k])) for k in numbers)), f'{name}: nonfinite number'
        tables[name] = rows
        checks[name] = {'rows': count, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': 'PASS'}
    seg, task, cal, errors = [tables[k] for k in specs]
    assert all((0 <= float(r['mean_dice']) <= 1 and float(r['fold_sd']) >= 0 and (int(r['folds']) == 5) for r in seg))
    assert all(((0 if r['endpoint_type'] == 'AUROC' else -1) <= float(r['value']) <= 1 for r in task))
    assert all((0 <= float(r['brier']) <= 1 for r in cal))
    assert sum((int(r['cases']) for r in errors)) == 489
    assert abs(sum((float(r['percentage']) for r in errors)) - 100) < 0.11
    assert all((abs(float(r['percentage']) - 100 * int(r['cases']) / 489) <= 0.051 for r in errors))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#35658B', '#216E64', '#805B19']

    def save(fig, name):
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(args.output_dir / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    architectures = list(dict.fromkeys((r['architecture'] for r in seg)))
    structures = list(dict.fromkeys((r['structure'] for r in seg)))
    for index, structure in enumerate(structures):
        rows = [r for r in seg if r['structure'] == structure]
        lookup = {r['architecture']: r for r in rows}
        ax.errorbar([float(lookup[a]['mean_dice']) for a in architectures], [i + (index - 1) * 0.18 for i in range(len(architectures))], xerr=[float(lookup[a]['fold_sd']) for a in architectures], fmt='o', capsize=3, label=structure, color=colors[index])
    ax.set(yticks=range(len(architectures)), yticklabels=architectures, xlim=(0, 1), xlabel='Dice (fold mean ± standard deviation)', title='Released segmentation summaries')
    ax.invert_yaxis()
    ax.legend(loc='lower left')
    save(fig, 'segmentation_summary')
    endpoints = list(dict.fromkeys((r['endpoint'] for r in task)))
    stages = list(dict.fromkeys((r['stage'] for r in task)))
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for index, stage in enumerate(stages):
        lookup = {r['endpoint']: r for r in task if r['stage'] == stage}
        ax.scatter([float(lookup[e]['value']) for e in endpoints], [i + (index - 1) * 0.18 for i in range(len(endpoints))], label=stage, color=colors[index])
    ax.set(yticks=range(len(endpoints)), yticklabels=endpoints, xlim=(0, 1), xlabel='QWK (ordinal tasks) / AUROC (binary tasks)', title='Released task performance')
    ax.invert_yaxis()
    ax.legend(loc='lower left')
    save(fig, 'task_performance')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout='constrained', sharey=True)
    for ax, metric, color in zip(axes, ['brier', 'slope'], colors):
        ax.scatter([float(r[metric]) for r in cal], range(len(cal)), color=color)
        ax.set(xlabel=metric.title(), yticks=range(len(cal)), yticklabels=[r['endpoint'] for r in cal])
        if metric == 'brier':
            ax.set_xlim(0, 0.3)
        else:
            ax.set_xlim(0, 1.5)
            ax.axvline(1, ls='--', color='0.6', lw=1)
    axes[0].invert_yaxis()
    fig.suptitle('Released calibration summaries')
    save(fig, 'calibration_summary')
    fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
    ax.barh([r['outcome'] for r in errors], [int(r['cases']) for r in errors], color=colors)
    for i, r in enumerate(errors):
        ax.text(int(r['cases']) + 4, i, f"{r['cases']} ({r['percentage']}%)", va='center')
    ax.set(xlabel='Cases (total 489)', xlim=(0, 370), title='Released Knosp error counts')
    ax.invert_yaxis()
    save(fig, 'knosp_error_summary')
    report = {'scope': 'aggregate_consistency_and_redrawing', 'datasets': checks, 'knosp_total': 489, 'numeric_inputs_modified': False, 'figure_count': 4}
    (args.output_dir / 'aggregate_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
if __name__ == '__main__':
    main()
