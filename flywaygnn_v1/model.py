from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.nn import GCNConv
except Exception as e:  # pragma: no cover
    raise ImportError("torch-geometric is required. Install with `pip install torch-geometric`.") from e


class DualChannelLayer(nn.Module):
    """Symmetric normalized dual-channel GCN with learnable mixture weights α (paper §4.3 style)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv_s = GCNConv(in_dim, out_dim, normalize=True, add_self_loops=False)
        self.conv_f = GCNConv(in_dim, out_dim, normalize=True, add_self_loops=False)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.raw_alpha = nn.Parameter(torch.zeros(2))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_s: torch.Tensor, edge_index_f: torch.Tensor) -> torch.Tensor:
        m_s = self.conv_s(x, edge_index_s)
        m_f = self.conv_f(x, edge_index_f)
        alpha = F.softmax(self.raw_alpha, dim=0)
        h = alpha[0] * m_s + alpha[1] * m_f + self.lin_self(x)
        h = self.act(h)
        h = self.drop(h)
        return h


class FlywayGNN(nn.Module):
    """H^(0)=MLP_enc(X), then two dual-channel GCN layers and a readout head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layer1 = DualChannelLayer(hidden_dim, hidden_dim, dropout)
        self.layer2 = DualChannelLayer(hidden_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index_s: torch.Tensor, edge_index_f: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = self.layer1(h, edge_index_s, edge_index_f)
        h = self.layer2(h, edge_index_s, edge_index_f)
        return self.head(h)


class VanillaSpatialGCN(nn.Module):
    """Baseline: single-channel GCN on spatial edges E_s only (no flyway edges E_f)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GCNConv(hidden_dim, hidden_dim, normalize=True, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=True, add_self_loops=False)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index_s: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = self.act(self.conv1(h, edge_index_s) + self.lin1(h))
        h = self.drop(h)
        h = self.act(self.conv2(h, edge_index_s) + self.lin2(h))
        h = self.drop(h)
        return self.head(h)
