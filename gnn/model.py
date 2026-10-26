"""Dense edge-conditioned attention GNN for tiny fully connected stone graphs.

Graphs have at most 13 nodes, so everything is batched as padded tensors with a node mask
instead of using a sparse graph library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, drop=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


class EdgeAttentionLayer(nn.Module):
    """GATv2-style layer: message and attention score both depend on (h_i, h_j, e_ij)."""

    def __init__(self, h, heads=4, drop=0.1):
        super().__init__()
        self.heads, self.dh = heads, h // heads
        self.msg = mlp(3 * h, h, h, drop)
        self.score = nn.Linear(h, heads)
        self.out = nn.Linear(h, h)
        self.norm1, self.norm2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = mlp(h, 2 * h, h, drop)
        self.edge_up = nn.Linear(h, h)

    def forward(self, h, e, mask):
        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        m = self.msg(torch.cat([hi, hj, e], -1))                         # [B,N,N,H]
        a = self.score(F.leaky_relu(m, 0.2))                             # [B,N,N,heads]
        pair = mask.unsqueeze(1) & mask.unsqueeze(2)
        pair = pair & ~torch.eye(N, dtype=torch.bool, device=h.device)
        a = a.masked_fill(~pair.unsqueeze(-1), -1e9).softmax(dim=2)
        a = a * pair.unsqueeze(-1)                                       # nodes with no neighbours
        mh = m.view(B, N, N, self.heads, self.dh)
        agg = (a.unsqueeze(-1) * mh).sum(2).reshape(B, N, H)
        h = self.norm1(h + self.out(agg))
        h = self.norm2(h + self.ff(h))
        e = e + self.edge_up(m)
        return h * mask.unsqueeze(-1), e


class NodeOnlyLayer(nn.Module):
    """Ablation: same capacity per node but no interaction between stones."""

    def __init__(self, h, drop=0.1):
        super().__init__()
        self.ff1, self.ff2 = mlp(h, 2 * h, h, drop), mlp(h, 2 * h, h, drop)
        self.norm1, self.norm2 = nn.LayerNorm(h), nn.LayerNorm(h)

    def forward(self, h, e, mask):
        h = self.norm1(h + self.ff1(h))
        h = self.norm2(h + self.ff2(h))
        return h * mask.unsqueeze(-1), e


class ShotGNN(nn.Module):
    """Graph encoder + readout [thrown-stone embedding | attention pool | context] -> ordinal logits."""

    def __init__(self, node_dim, edge_dim, ctx_dim, hidden=64, layers=3, use_edges=True,
                 n_classes=5, drop=0.1):
        super().__init__()
        self.use_edges = use_edges
        self.ctx_enc = mlp(ctx_dim, hidden, hidden, drop)
        self.node_enc = mlp(node_dim + hidden, hidden, hidden, drop)    # task/context-conditioned nodes
        self.edge_enc = mlp(edge_dim, hidden, hidden, drop) if use_edges else None
        self.layers = nn.ModuleList(
            [EdgeAttentionLayer(hidden, drop=drop) if use_edges else NodeOnlyLayer(hidden, drop)
             for _ in range(layers)])
        self.pool_score = nn.Linear(hidden, 1)
        self.head = mlp(3 * hidden, hidden, n_classes - 1, drop)        # K-1 cumulative logits

    def forward(self, node, edge, mask, thrown, ctx):
        B, N, _ = node.shape
        c = self.ctx_enc(ctx)
        h = self.node_enc(torch.cat([node, c.unsqueeze(1).expand(B, N, -1)], -1)) * mask.unsqueeze(-1)
        e = self.edge_enc(edge) if self.use_edges else None
        for layer in self.layers:
            h, e = layer(h, e, mask)
        w = self.pool_score(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        pooled = (w.unsqueeze(-1) * h).sum(1)
        idx = thrown.clamp(min=0)
        th = h[torch.arange(B, device=h.device), idx] * (thrown >= 0).float().unsqueeze(-1)
        return self.head(torch.cat([th, pooled, c], -1))


def ordinal_loss(logits, y):
    """Cumulative-link ordinal loss: independent BCE on each threshold P(y > k)."""
    k = torch.arange(logits.shape[1], device=y.device)
    target = (y.unsqueeze(1) > k).float()
    return F.binary_cross_entropy_with_logits(logits, target)


def ordinal_probs(logits):
    """Class probabilities from cumulative logits (made monotone first)."""
    p_gt = torch.sigmoid(logits)
    p_gt = torch.cummin(p_gt, dim=1).values
    ones = torch.ones_like(p_gt[:, :1])
    zeros = torch.zeros_like(p_gt[:, :1])
    cum = torch.cat([ones, p_gt, zeros], 1)
    return (cum[:, :-1] - cum[:, 1:]).clamp(min=0)
