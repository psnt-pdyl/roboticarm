"""Motion PINN: joint angles as a function of time for a rest-to-rest move.

    q(s) = q0 + (q1 - q0) b(s) + w(s) N(q0, q1 - q0, s, T, payload),   s = t / T in [0, 1]

b(s) = 10s^3 - 15s^4 + 6s^5 and w(s) = 64 s^3 (1 - s)^3 make the start and end exact by
construction: right poses, zero velocity and acceleration at both ends. The network N
only shapes the path in between, and it is trained on physics alone (see train.py).
"""
import math

import torch
from torch import nn

from dynamics import LIM

Q_MID = LIM.mean(1)
Q_HALF = (LIM[:, 1] - LIM[:, 0]) / 2
T_REF = 1.0          # s; the network sees log(T / T_REF)


def basis(s):
    """b(s), w(s) and their first and second derivatives with respect to s."""
    b = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    bd = 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4
    bdd = 60 * s - 180 * s ** 2 + 120 * s ** 3
    w = 64 * s ** 3 * (1 - s) ** 3
    wd = 64 * (3 * s ** 2 * (1 - s) ** 3 - 3 * s ** 3 * (1 - s) ** 2)
    wdd = 64 * (6 * s * (1 - s) ** 3 - 18 * s ** 2 * (1 - s) ** 2 + 6 * s ** 3 * (1 - s))
    return (b, bd, bdd), (w, wd, wdd)


class MotionPINN(nn.Module):
    def __init__(self, hidden=256, layers=4):
        super().__init__()
        dims = [9] + [hidden] * layers
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims, dims[1:]))
        self.out = nn.Linear(hidden, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                  # start as the plain quintic blend
        self.register_buffer("q_mid", Q_MID.clone())
        self.register_buffer("q_half", Q_HALF.clone())

    def features(self, q0, q1, s, T, payload):
        return torch.cat([(q0 - self.q_mid) / self.q_half, (q1 - q0) / self.q_half,
                          (2 * s - 1)[..., None], (torch.log(T / T_REF))[..., None],
                          payload[..., None]], -1)

    def shape(self, q0, q1, s, T, payload):
        """The network's path correction N [..., 3] (radians, at the middle of the move)."""
        x = self.features(q0, q1, s, T, payload)
        for layer in self.layers:
            x = nn.functional.silu(layer(x))
        return self.out(x)

    def forward(self, q0, q1, s, T, payload):
        """q(s) [..., 3]. q0, q1 [..., 3]; s, T (seconds), payload (kg) [...]."""
        (b, _, _), (w, _, _) = basis(s)
        return q0 + (q1 - q0) * b[..., None] + w[..., None] * self.shape(q0, q1, s, T, payload)

    def motion(self, q0, q1, s, T, payload):
        """q, qd, qdd in physical time (rad, rad/s, rad/s^2). Derivatives by autograd through s."""
        return compose(lambda s_: self.shape(q0, q1, s_, T, payload), q0, q1, s, T)


def compose(shape_fn, q0, q1, s, T):
    """q, qd, qdd for q(s) = q0 + (q1 - q0) b(s) + w(s) shape_fn(s), with s = t / T."""
    s = s.detach().requires_grad_(True)
    N = shape_fn(s)
    dN = torch.stack([torch.autograd.grad(N[..., j].sum(), s, create_graph=True)[0] for j in range(3)], -1)
    ddN = torch.stack([torch.autograd.grad(dN[..., j].sum(), s, create_graph=True)[0] for j in range(3)], -1)
    (b, bd, bdd), (w, wd, wdd) = basis(s)
    dq = q1 - q0
    q = q0 + dq * b[..., None] + w[..., None] * N
    q_s = dq * bd[..., None] + wd[..., None] * N + w[..., None] * dN
    q_ss = dq * bdd[..., None] + wdd[..., None] * N + 2 * wd[..., None] * dN + w[..., None] * ddN
    Tt = T[..., None]
    return q, q_s / Tt, q_ss / Tt ** 2
