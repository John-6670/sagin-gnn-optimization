from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class SAGINHeteroGNN(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=128, layers=3, heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.encoders = nn.ModuleDict({k: nn.Linear(v, hidden_dim) for k, v in in_dims.items()})
        self.convs = nn.ModuleList()
        for _ in range(layers):
            conv_dict = {}
            for etype in metadata[1]:
                conv_dict[etype] = GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
        self.heads = nn.ModuleDict({
            "satellite": nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            "hap": nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            "ground": nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            # optional safety for missing types
            "client": nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        })

    def forward(self, data, placement_mask):
        x_dict = {k: F.relu(self.encoders[k](v)) for k, v in data.x_dict.items()}
        for conv in self.convs:
            out_dict = conv(x_dict, data.edge_index_dict)

            for node_type in x_dict:
                if node_type not in out_dict:
                    out_dict[node_type] = x_dict[node_type]

            x_dict = out_dict
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}

        all_embeddings = torch.cat([x_dict[k] for k in x_dict], dim=0)
        g_s = all_embeddings[placement_mask].mean(dim=0, keepdim=True) if placement_mask.any() else all_embeddings.mean(dim=0, keepdim=True)
        pred = {}
        for k, emb in x_dict.items():
            ctx = g_s.repeat(emb.size(0), 1)
            h = torch.cat([emb, ctx], dim=-1)

            if k in self.heads:
                pred[k] = F.relu(self.heads[k](h).squeeze(-1))
            else:
                pred[k] = F.relu(self.heads["ground"](h).squeeze(-1))

        return pred


class SubmodularityRegularizer(nn.Module):
    def forward(self, gains_a, gains_b):
        """
        Submodularity regularizer per Eq. 35.
        For A subset B and v in B minus A: gain(A, v) >= gain(B, v)

        Args:
            gains_a: Marginal gains for smaller set A
            gains_b: Marginal gains for larger set B (containing A)

        Returns:
            Mean violation: max(0, gain(B, v) - gain(A, v))
        """
        return F.relu(gains_b - gains_a).mean()


class SNRGradientPenalty(nn.Module):
    def forward(self, preds, snr_feature):
        """
        SNR gradient penalty per Eq. 35.
        Encourages negative correlation: d(pred)/d(SNR) < 0
        (higher SNR -> less need for this server as relay)

        Args:
            preds: Model predictions for candidates
            snr_feature: SNR feature values (not noise_variance)

        Returns:
            Mean squared gradient magnitude
        """
        grads = torch.autograd.grad(preds.sum(), snr_feature, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if grads is None:
            return torch.tensor(0.0, device=preds.device)
        # Encourage negative gradient (higher SNR -> lower marginal gain)
        return (grads ** 2).mean()
