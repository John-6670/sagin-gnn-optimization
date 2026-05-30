import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_dro_results(name):
    files = glob.glob(f'results/{name}')
    if not files:
        print(f"No DRO {name} results found in results/")
        return None
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(os.path.basename(f))
        tag = os.path.basename(f).replace('dr_sensitivity_', '').replace('_metrics.csv', '').replace('dr_ablation_', '')
        print(f"  tag={tag}")
        df['variant'] = tag
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_sensitivity_dro():
    os.makedirs('plots/sensitivity_dro', exist_ok=True)
    df = load_dro_results('dr_sensitivity_*_metrics.csv')
    if df is None:
        return

    metrics = ['latency', 'amse', 'energy', 'cvar5']
    hyperparameters = {
        'Budget': ['budget_20', 'budget_30', 'budget_40'],
        'N (Scenarios)': ['N_32', 'N_64'],
        'Epsilon': ['eps_005', 'eps_015'],
        'Kappa': ['kappa_015', 'kappa_030']
    }

    for metric in metrics:
        fig, axs = plt.subplots(2, 2, figsize=(18, 13))
        axs = axs.flatten()
        
        for i, (param_name, variants) in enumerate(hyperparameters.items()):
            ax = axs[i]
            for variant in variants:
                sub = df[df['variant'] == variant]
                if sub.empty:
                    continue
                    
                grouped = sub.groupby('step')[metric].mean()
                std = sub.groupby('step')[metric].std().fillna(0)
                
                label = variant.replace('_', '=').replace('budget', 'B').replace('N_', 'N=').replace('eps_', 'ε=').replace('kappa_', 'κ=')
                
                # Special handling for AMSE to make differences more visible
                if metric == 'amse':
                    # Use log scale for AMSE to better show relative differences
                    ax.semilogy(grouped.index, grouped.values, 
                               label=label, linewidth=1, marker='o', markersize=1.5)
                    ax.fill_between(grouped.index, 
                                   grouped.values * 0.85,  # tighter band for visibility
                                   grouped.values * 1.15, alpha=0.25)
                else:
                    ax.plot(grouped.index, grouped.values, 
                           label=label, linewidth=2, marker='o', markersize=2.5)
                    ax.fill_between(grouped.index, 
                                   grouped.values - std, 
                                   grouped.values + std, alpha=0.2)
            
            ax.set_title(f'Varying {param_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Simulation Step')
            ax.set_ylabel(metric.capitalize() + (' (log scale)' if metric == 'amse' else ''))
            ax.legend(title="Parameter Value", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'DRO Sensitivity Analysis - {metric.upper()}', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = f'plots/sensitivity_dro/sensitivity_{metric}_multi.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved: {filename}")

def plot_ablation_dro():
    os.makedirs('plots/ablation_dro', exist_ok=True)
    df = load_dro_results("dr_ablation_*_metrics.csv")  # reuse function with pattern
    if df is None:
        return

    summary = df.groupby('variant')[['latency', 'amse', 'energy', 'cvar5']].mean()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    metrics = ['latency', 'amse', 'energy', 'cvar5']
    
    for i, metric in enumerate(metrics):
        summary[metric].plot(kind='bar', ax=axs[i], color='steelblue', edgecolor='black')
        axs[i].set_title(f'Ablation: {metric.upper()}', fontsize=13, fontweight='bold')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].grid(axis='y', alpha=0.3)
        axs[i].set_ylim(0, summary[metric].max() * 1.5)
        
        for p in axs[i].patches:
            axs[i].text(p.get_x() + p.get_width()/2., p.get_height() + 0.005,
                       f'{p.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('DRO Ablation Study - Impact of Components', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/ablation_dro/ablation_summary.png', dpi=250, bbox_inches='tight')
    print("✅ Ablation plot saved.")

if __name__ == "__main__":
    plot_sensitivity_dro()
    plot_ablation_dro()