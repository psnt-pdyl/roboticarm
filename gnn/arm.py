"""Arm geometry and kinematics, matching the web simulation (index.html) exactly.

Units: millimetres and radians. Angles are relative joint angles; the tool angle
phi is the absolute direction of the last link (q1 + q2 + q3).
"""
import math

import torch

L1, L2, L3 = 360.0, 300.0, 120.0
SH = (0.0, 160.0)                       # shoulder position
REACH = L1 + L2 + L3                    # 780 mm max TCP reach from the shoulder
D = math.pi / 180
LIM = torch.tensor([[-20.0, 200.0], [-160.0, 160.0], [-140.0, 140.0]]) * D
Q_MID = LIM.mean(1)                     # output de-normalisation: q = mid + half * y
Q_HALF = (LIM[:, 1] - LIM[:, 0]) / 2


def fk(q):
    """Forward kinematics. q: [..., 3] -> (tcp [..., 2], tool angle [...])."""
    a1 = q[..., 0]
    a2 = a1 + q[..., 1]
    a3 = a2 + q[..., 2]
    x = SH[0] + L1 * torch.cos(a1) + L2 * torch.cos(a2) + L3 * torch.cos(a3)
    y = SH[1] + L1 * torch.sin(a1) + L2 * torch.sin(a2) + L3 * torch.sin(a3)
    return torch.stack([x, y], -1), a3


def sample(n, device="cpu", generator=None):
    """Uniform joint configurations inside the limits, with the targets they reach.

    Returns q [n,3], tcp [n,2], phi [n], elbow [n] (+1 / -1). The elbow flag picks
    which of the two IK branches is wanted, which makes target -> angles a function.
    """
    lim = LIM.to(device)
    u = torch.rand(n, 3, device=device, generator=generator)
    q = lim[:, 0] + u * (lim[:, 1] - lim[:, 0])
    tcp, phi = fk(q)
    elbow = torch.where(q[:, 1] > 0, 1.0, -1.0)
    return q, tcp, phi, elbow


def goal_features(tcp, phi, elbow):
    """Goal-node input: shoulder-relative target scaled by reach, tool angle as sin/cos, elbow branch."""
    return torch.stack([
        (tcp[..., 0] - SH[0]) / REACH,
        (tcp[..., 1] - SH[1]) / REACH,
        torch.sin(phi),
        torch.cos(phi),
        elbow,
    ], -1)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def analytic_ik(tcp, phi, elbow):
    """Closed-form IK used by the web page, for reference and sanity checks."""
    wx = tcp[..., 0] - L3 * torch.cos(phi) - SH[0]
    wy = tcp[..., 1] - L3 * torch.sin(phi) - SH[1]
    d2 = wx * wx + wy * wy
    c = ((d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)).clamp(-1, 1)
    q2 = elbow * torch.acos(c)
    q1 = wrap(torch.atan2(wy, wx) - torch.atan2(L2 * torch.sin(q2), L1 + L2 * torch.cos(q2)))
    q1 = torch.where(q1 < -90 * D, q1 + 2 * math.pi, q1)
    q3 = wrap(phi - q1 - q2)
    return torch.stack([q1, q2, q3], -1)
