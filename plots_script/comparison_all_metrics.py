import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


files = glob.glob('results/*_metrics.csv')
if not files:
    raise SystemExit('no result csv')

metrics = ['latency', 'amse', 'energy', 'cvar5']
fig, axs = plt.subplots(2, 3, figsize=(16,8))
axs = axs.flatten()
for f in files:
    name = os.path.basename(f).replace('_metrics.csv', '')
    df = pd.read_csv(f)
    x = df['step']
    for i, m in enumerate(metrics):
        y = df[m].values
        mu = y
        std = np.zeros_like(y)
        axs[i].plot(x, mu, label=name)
        axs[i].fill_between(x, mu-1.96*std, mu+1.96*std, alpha=0.1)
    
    axs[4].plot(x, 1/(1+df['amse'].values), label=name)

for i, t in enumerate(['Latency', 'AMSE', 'Energy', 'CVaR@5%', 'Convergence (proxy)', '']):
    axs[i].set_title(t)

for i in range(5):
    axs[i].legend(fontsize=7)

axs[5].axis('off')
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/comparison_all_metrics.png', dpi=200)
print('saved plots/comparison_all_metrics.png')
