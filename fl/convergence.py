from __future__ import annotations
import numpy as np


def convergence_monitor(amse_history, empirical_loss_history, sigma2, rho, gamma, f0_minus_fstar=1.0):
    bounds=[]
    logs=[]
    for t, amse in enumerate(amse_history, start=1):
        b = (rho**t)*f0_minus_fstar + (sigma2*(1+gamma*amse))/max(1e-9, (1-rho))
        bounds.append(float(b))
        logs.append({
                        'round': t,
                        'amse': float(amse),
                        'empirical_loss': float(empirical_loss_history[t-1]),
                        'theoretical_bound': float(b)    
                    })
    return np.array(bounds,dtype=float), logs
