from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MAMLInnerOptimizer:
    def __init__(self, param_dim: int = 2, inner_lr: float = 1e-2, meta_lr: float = 5e-3, inner_steps: int = 5):
        self.phi = torch.nn.Parameter(torch.zeros(param_dim, dtype=torch.float32))
        self.inner_lr = inner_lr
        self.meta_opt = torch.optim.Adam([self.phi], lr=meta_lr)
        self.inner_steps = inner_steps
        self.sync_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.sync_opt = torch.optim.Adam(self.sync_mlp.parameters(), lr=meta_lr)

    def _loss(self, snr_tensor, params):
        power_scale = torch.sigmoid(params[0])
        phase_bias = params[1]
        effective = snr_tensor * power_scale * torch.cos(phase_bias).abs().clamp(min=1e-3)
        return torch.mean(1.0 / effective.clamp(min=1e-6))

    def adapt(self, snr_values):
        snr_t = torch.tensor(np.asarray(snr_values, dtype=np.float32) + 1e-6)
        params = self.phi.clone().requires_grad_(True)
        for _ in range(self.inner_steps):
            loss = self._loss(snr_t, params)
            grad = torch.autograd.grad(loss, params, create_graph=True)[0]
            params = params - self.inner_lr * grad
        return params, self._loss(snr_t, params)

    def maml_update(self, snr_values, traffic, mobility):
        self.meta_opt.zero_grad()
        self.sync_opt.zero_grad()
        adapted, meta_loss = self.adapt(snr_values)
        snr_mean = float(np.mean(snr_values)) if len(snr_values) else 0.0
        sync_adj = self.sync_mlp(torch.tensor([traffic, mobility, snr_mean], dtype=torch.float32))
        total_loss = meta_loss + 0.01 * sync_adj.pow(2).mean()
        total_loss.backward()
        self.meta_opt.step()
        self.sync_opt.step()
        return adapted.detach(), float(sync_adj.item()), float(total_loss.item())
