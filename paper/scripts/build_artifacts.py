"""Build descriptive paper artifacts from existing logs; never trains or rewrites logs."""
from pathlib import Path
import csv
import hashlib
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'
sys.path.insert(0, str(ROOT / 'src'))
from summarize import read, honest_rows, compared_bill, ci, DEFENCES, DATA_ATTACKS

paths = sorted((ROOT / 'results').glob('*_sac_*.log'))
runs = pd.DataFrame([read(p) for p in paths])
runs['seed'] = runs.seed.astype(int)
runs[['attack', 'share', 'variant']] = runs[['attack', 'share', 'variant']].fillna('')
attacks = ['flip', 'scale', 'noise', 'normalized', 'meter', 'reward']
metrics, damage_rows = [], []
for last in (2, 4):
    selected = runs[runs.seed.le(last) & runs.share.eq('30')]
    main = selected[selected.variant.eq('')].copy()
    reference = honest_rows(main, selected[selected.variant.eq('honest')])
    none = main[main.attack.eq('none')]
    all_ref = none.set_index(['method', 'seed'])['bill $/home/yr'].to_dict()
    honest_ref = reference.set_index(['method', 'seed'])['honest bill'].to_dict()
    hit = main[main.attack.isin(attacks)].copy()
    hit['damage'] = compared_bill(hit) - np.array([
        (honest_ref if a in DATA_ATTACKS else all_ref).get((m, s), np.nan)
        for m, s, a in zip(hit.method, hit.seed, hit.attack)])
    for method in DEFENCES:
        part = hit[hit.method.eq(method)]
        if part.empty:
            continue
        counts = part.groupby('attack').seed.nunique().reindex(attacks, fill_value=0)
        if not counts.eq(last + 1).all() or part.damage.isna().any():
            raise ValueError(f'Incomplete damage coverage: {method}, seeds 0-{last}')
        means = part.groupby('attack').damage.mean()
        worst = means.idxmax()
        rejected = part.groupby('attack')['honest rejected %'].mean()
        metrics.append(dict(method=method, seeds=f'0-{last}', worst_attack=worst,
                            worst_mean_damage=means[worst],
                            mean_logged_honest_rejection=rejected.mean(),
                            rejection_observations=int(part['honest rejected %'].count()),
                            expected_observations=6*(last+1)))
        if last == 4:
            damage_rows.append((method, [ci(part.loc[part.attack.eq(a), 'damage']) for a in attacks]))
frame = pd.DataFrame(metrics)
frame.to_csv(PAPER / 'tables/damage_rejection.csv', index=False)
# Preserve the exact plotted inputs alongside a hash manifest of every summary log.
with (PAPER / 'tables/source_logs.sha256').open('w') as out:
    for path in paths:
        out.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}\n')
lines = [r'\begin{table}[ht]', r'\centering\scriptsize',
         r'\caption{Ausgrid annual bill damage, seeds 0--4. Entries are mean $\pm$ 95\% confidence interval half-width [seeds], in dollars per household. These are intervals for each attack, not for the maximum across attacks.}',
         r'\label{tab:ausgrid_damage}', r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{lrrrrrr}', r'\toprule',
         r'Method & Flip & Scale & Noise & Normalized & Meter & Reward \\', r'\midrule']
for method, values in damage_rows:
    cells = [v.replace('±', r'$\pm$') for v in values]
    lines.append(method.replace('fedgrid', 'FedGrid').replace('fltrust', 'FLTrust').replace('flame', 'FLAME').replace('tafdrl', 'TAFDRL-style').replace('fedavg', 'FedAvg').replace('krum', 'Krum') + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
(PAPER / 'tables/main_damage.tex').write_text('\n'.join(lines) + '\n')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('Tables and provenance built. Install matplotlib in the separate paper environment to render the figure.')
    raise SystemExit(0)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), sharey=True, layout='constrained')
for ax, label in zip(axes, ('0-2', '0-4')):
    part = frame[frame.seeds.eq(label)]
    for row in part.itertuples():
        if pd.isna(row.mean_logged_honest_rejection):
            continue
        final = row.method == 'fedgrid final'
        ax.scatter(row.mean_logged_honest_rejection, row.worst_mean_damage,
                   color='#b23b26' if final else '#235789', s=65 if final else 30,
                   marker='D' if final else 'o')
        ax.annotate(row.method, (row.mean_logged_honest_rejection, row.worst_mean_damage),
                    xytext=(4, 5 if row.method != 'fltrust' else -12),
                    textcoords='offset points', fontsize=7)
    ax.set_title(f'Ausgrid, seeds {label}')
    ax.set_xlabel('Mean logged honest rejection across six attacks (%)')
    ax.set_xlim(-4, 119)
    ax.grid(alpha=.2)
axes[0].set_ylabel('Maximum mean annual bill damage across attacks ($/home)')
fig.suptitle('Attack damage and recorded rejection\nAdmission definitions differ by defence; effective exclusion is not measured', fontsize=11)
fig.savefig(PAPER / 'figures/damage_rejection.pdf')
fig.savefig(PAPER / 'figures/damage_rejection.png', dpi=220)
plt.close(fig)
print('Built damage table, plot inputs, log hashes and PDF/PNG figure.')
