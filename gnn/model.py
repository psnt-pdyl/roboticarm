"""Graph network for inverse kinematics.

The graph is the arm itself plus the goal:

    shoulder(0) -- elbow(1) -- wrist(2) -- tcp(3)
         \\           |           |          /
          '--------- goal(4) ---------------'

Chain edges carry the link length. The goal node carries the target. After a few
rounds of message passing, each joint node (0, 1, 2) decodes its own angle.
Written in plain PyTorch (no PyG) so the forward pass is easy to port to JavaScript.
"""
import torch
from torch import nn

from arm import L1, L2, L3, REACH, Q_MID, Q_HALF

N_NODES = 5
GOAL = 4
# (src, dst, edge type, link length / reach)
_EDGES = []
for (a, b), length in zip([(0, 1), (1, 2), (2, 3)], [L1, L2, L3]):
    _EDGES.append((a, b, 0, length / REACH))   # chain, outward
    _EDGES.append((b, a, 1, length / REACH))   # chain, inward
for i in range(4):
    _EDGES.append((GOAL, i, 2, 0.0))            # goal -> joint
    _EDGES.append((i, GOAL, 3, 0.0))            # joint -> goal
SRC = [e[0] for e in _EDGES]
DST = [e[1] for e in _EDGES]
N_EDGES = len(_EDGES)
NODE_IN = N_NODES + 5     # one-hot node type + goal features (zeros except on the goal node)
EDGE_IN = 4 + 1           # one-hot edge type + link length


def static_features():
    node = torch.zeros(N_NODES, NODE_IN)
    node[torch.arange(N_NODES), torch.arange(N_NODES)] = 1.0
    edge = torch.zeros(N_EDGES, EDGE_IN)
    for k, (_, _, t, length) in enumerate(_EDGES):
        edge[k, t] = 1.0
        edge[k, 4] = length
    return node, edge


class MLP(nn.Module):
    """Linear -> SiLU -> Linear, optionally followed by LayerNorm."""

    def __init__(self, i, h, o, norm=True):
        super().__init__()
        self.l1 = nn.Linear(i, h)
        self.l2 = nn.Linear(h, o)
        self.norm = nn.LayerNorm(o) if norm else None

    def forward(self, x):
        x = self.l2(nn.functional.silu(self.l1(x)))
        return self.norm(x) if self.norm is not None else x


class IKGNN(nn.Module):
    def __init__(self, hidden=96, steps=5):
        super().__init__()
        self.hidden, self.steps = hidden, steps
        node, edge = static_features()
        self.register_buffer("node0", node)
        self.register_buffer("edge0", edge)
        self.register_buffer("src", torch.tensor(SRC))
        self.register_buffer("dst", torch.tensor(DST))
        self.register_buffer("q_mid", Q_MID.clone())
        self.register_buffer("q_half", Q_HALF.clone())
        self.enc_node = MLP(NODE_IN, hidden, hidden)
        self.enc_edge = MLP(EDGE_IN, hidden, hidden)
        self.edge_mlps = nn.ModuleList(MLP(3 * hidden, hidden, hidden) for _ in range(steps))
        self.node_mlps = nn.ModuleList(MLP(2 * hidden, hidden, hidden) for _ in range(steps))
        self.dec = MLP(hidden, hidden, 1, norm=False)

    def forward(self, goal):
        """goal: [B, 5] goal features -> joint angles q [B, 3] in radians."""
        B = goal.shape[0]
        x = self.node0.expand(B, -1, -1).clone()
        x[:, GOAL, N_NODES:] = goal
        h = self.enc_node(x)                                   # [B, N, H]
        e = self.enc_edge(self.edge0).expand(B, -1, -1)        # [B, E, H]
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e = e + edge_mlp(torch.cat([e, h[:, self.src], h[:, self.dst]], -1))
            agg = torch.zeros_like(h).index_add_(1, self.dst, e)
            h = h + node_mlp(torch.cat([h, agg], -1))
        y = self.dec(h[:, :3]).squeeze(-1)                     # joints 0..2
        # tanh keeps each angle inside its joint limits (all under 360 deg wide), so the
        # network can't settle on a 2*pi-shifted equivalent that FK can't tell apart.
        return self.q_mid + self.q_half * torch.tanh(y)
