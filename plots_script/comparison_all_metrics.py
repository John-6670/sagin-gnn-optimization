import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


files = glob.glob('results/*_metrics.csv')
if not files:
    raise SystemExit('no result csv files found in results/')

metrics = ['latency', 'amse', 'energy', 'cvar5', 'cvar95', 'cvar90', 'cvar99', 'num_sat', 'num_uav', 'num_ground']
KNOWN_ALGOS = ["da", "lop", "go", "nrs", "dr", "random", "fedsn", "hsfl"]
DISPLAY_NAME = {
    'dr_greedy': 'proposed_dr_greedy',
    'greedy': 'greedy',
    'lop': 'lop',
    'go': 'go',
    'nrs': 'nrs',
    'random': 'random',
    'da': 'da',
    'fedsn': 'fedsn',
    'hsfl': 'hsfl',
}


def parse_algo_and_run(base_name: str):
    """Parse `algo` and `run tag` from <algo>[_<tag>] filename stem.

    This is robust to algo names containing underscores (e.g., dr_greedy).
    """
    for algo in sorted(KNOWN_ALGOS, key=len, reverse=True):
        if base_name == algo:
            return algo, 'run0'
        if base_name.startswith(algo + '_'):
            return algo, base_name[len(algo) + 1 :]
    return None, None


rows = []
for f in files:
    base = os.path.basename(f).replace('_metrics.csv', '')
    algo, run_tag = parse_algo_and_run(base)
    if algo is None:
        print(f'skipping unrecognized metrics file: {base}')
        continue
    
    df = pd.read_csv(f)
    df['algo'] = algo
    df['run'] = run_tag
    rows.append(df)
    
if not rows:
    raise SystemExit('no recognized result csvs after parsing')

all_df = pd.concat(rows, ignore_index=True)

fig, axs = plt.subplots(5, 2, figsize=(16, 25))
axs = axs.flatten()

algorithms = sorted(all_df['algo'].unique(), key=lambda x: KNOWN_ALGOS.index(x))

# For AMSE, we'll handle it separately with better scaling
for algo in algorithms:
    df_a = all_df[all_df['algo'] == algo]
    grouped = df_a.groupby('step')
    
    for i, m in enumerate(metrics):
        if m not in all_df.columns:
            continue
        mu = grouped[m].mean().sort_index()
        std = grouped[m].std(ddof=1).fillna(0.0).reindex(mu.index)
        n = grouped[m].count().reindex(mu.index).clip(lower=1)
        ci = 1.96 * std / np.sqrt(n)

        if i == 2:  # energy
            axs[i].plot(mu.index.values, mu.values, label=DISPLAY_NAME.get(algo, algo), linewidth=1)
            axs[i].fill_between(mu.index.values, (mu - ci).values, (mu + ci).values, alpha=0.12, linewidth=1)
            # axs[i].set_ylim(0, mu.max() * 1.1 if not np.isnan(mu.max()) else 1)
        elif i == 1:  # amse
            axs[i].plot(mu.index.values, mu.values, label=DISPLAY_NAME.get(algo, algo), linewidth=0.5)
            axs[i].fill_between(mu.index.values, np.clip((mu - ci).values, 1e-6, None), np.clip((mu + ci).values, 1e-6, None), alpha=0.12, linewidth=1)
            axs[i].set_yscale('log')
        elif i == 0:
            axs[i].plot(mu.index.values, mu.values, label=DISPLAY_NAME.get(algo, algo), linewidth=1)
            axs[i].fill_between(mu.index.values, (mu - ci).values, (mu + ci).values, alpha=0.12, linewidth=1)
            # axs[i].set_ylim(0, mu.max() * 0.2)
        elif i > 6:  # latency or device counts
            axs[i].plot(mu.index.values, mu.values, label=DISPLAY_NAME.get(algo, algo), linewidth=1)
            axs[i].fill_between(mu.index.values, (mu - ci).values, (mu + ci).values, alpha=0.12, linewidth=1)
        else:
            # CVaR (log scale)
            axs[i].plot(mu.index.values, mu.values, label=DISPLAY_NAME.get(algo, algo), linewidth=1)
            axs[i].fill_between(mu.index.values, np.clip((mu - ci).values, 1e-6, None), 
                               np.clip((mu + ci).values, 1e-6, None), alpha=0.12)
            axs[i].set_yscale('log')


titles = ['Latency', 'AMSE', 'Energy', 'CVaR@5%', 'CVaR@95%', 'CVaR@90%', 'CVaR@99%', '# Satellites', '# UAVs', '# Ground BS']
for i, t in enumerate(titles):
    axs[i].set_title(t, fontsize=11, fontweight='bold')

for i in range(len(metrics)):
    handles, labels = axs[i].get_legend_handles_labels()
    if handles:
        axs[i].legend(fontsize=8, loc='best')
    axs[i].set_xlabel('step', fontsize=9)
    axs[i].grid(True, alpha=0.3)
    if i != 1 and not (3 <= i <= 6):  # Don't set ylabel for log scale (it's automatic)
        axs[i].set_ylabel('value', fontsize=9)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/comparison_all_metrics.png', dpi=200, bbox_inches='tight')
plt.close()

valid_metrics = [m for m in metrics if m in all_df.columns]
summary = all_df.groupby('algo')[valid_metrics].mean().sort_values('amse')
summary.index = [DISPLAY_NAME.get(idx, idx) for idx in summary.index]
summary.to_csv('results/summary_metrics.csv')
print('saved plots/comparison_all_metrics.png')
print('saved results/summary_metrics.csv')
