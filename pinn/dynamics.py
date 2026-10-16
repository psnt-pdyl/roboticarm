"""Rigid-body dynamics of the planar arm, plus the motion the page uses today.

The page's arm is kinematic: it has joint speed and acceleration limits but no masses.
This module gives it a body: link masses, inertias, gravity and motor torque limits,
and computes the torque each motor needs for a given motion (inverse dynamics).

Units here are SI (m, kg, s, rad, N m). Lengths match gnn/arm.py.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "gnn"))
from arm import L1, L2, L3, LIM, SH  # noqa: E402  (mm, radians)

G = 9.81
LEN = torch.tensor([L1, L2, L3]) / 1000          # link lengths, m
MASS = torch.tensor([6.0, 4.0, 1.5])             # link masses, kg (link 3: wrist, tool and gripper)
COM = LEN / 2                                     # each link's centre of mass sits mid-length
INERTIA = MASS * LEN ** 2 / 12                    # about the centre of mass, kg m^2 (uniform rod)
TAU_MAX = torch.tensor([80.0, 30.0, 6.0])         # motor torque limits, N m
PART_MASS = 0.4                                   # one part, kg (carried at the tool point)
D = math.pi / 180
VMAX = torch.tensor([110.0, 130.0, 200.0]) * D    # the page's joint speed limits at 100 % override
AMAX = torch.tensor([320.0, 380.0, 600.0]) * D    # and acceleration limits
FLOOR_CLEAR = 20.0                                # mm: elbow, wrist and tool point stay this far above the floor
JOINT_MARGIN = 1.0 * D                            # motions keep this far inside the joint limits


def inverse_dynamics(q, qd, qdd, payload):
    """Motor torques [..., 3] for joint angles, velocities, accelerations [..., 3] and payload mass [...].

    Planar recursive Newton-Euler: accelerations go out from the fixed shoulder, forces
    and moments come back in from the tool point, where the payload hangs as a point mass.
    """
    dev = q.device
    L, m, c, I = (t.to(dev) for t in (LEN, MASS, COM, INERTIA))
    a, ad, add = q.cumsum(-1), qd.cumsum(-1), qdd.cumsum(-1)        # absolute link angles and rates
    u = torch.stack([a.cos(), a.sin()], -1)                         # along each link
    n = torch.stack([-a.sin(), a.cos()], -1)                        # perpendicular to it
    rel = add[..., None] * n - (ad ** 2)[..., None] * u             # acceleration per metre along the link
    g = torch.tensor([0.0, -G], device=dev)
    cross = lambda r, f: r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]

    p_acc = [torch.zeros_like(u[..., 0, :])]                         # joint-origin accelerations
    for k in range(3):
        p_acc.append(p_acc[-1] + L[k] * rel[..., k, :])
    F = payload[..., None] * (p_acc[3] - g)                          # force on the payload
    tau_next = torch.zeros_like(q[..., 0])
    taus = [None] * 3
    for k in (2, 1, 0):
        c_acc = p_acc[k] + c[k] * rel[..., k, :]
        F_k = m[k] * (c_acc - g) + F                                 # force the joint passes into link k
        taus[k] = (I[k] * add[..., k] + tau_next
                   + c[k] * cross(u[..., k, :], F_k) + (L[k] - c[k]) * cross(u[..., k, :], F))
        F, tau_next = F_k, taus[k]
    return torch.stack(taus, -1)


def points_z(q):
    """Height above the floor (mm) of the elbow, wrist and tool point [..., 3]."""
    a = q.cumsum(-1)
    z = SH[1] + (torch.tensor([L1, L2, L3], device=q.device) * a.sin()).cumsum(-1)
    return z


def move_time(q0, q1, override=1.0):
    """Duration of the page's joint move (each joint its own speed/acceleration trapezoid) [...]."""
    v, acc = VMAX.to(q0.device) * override, AMAX.to(q0.device) * override
    d = (q1 - q0).abs()
    t = torch.where(d >= v * v / acc, d / v + v / acc, 2 * torch.sqrt(d / acc))
    return t.max(-1).values


def blend(q0, q1, t, override=1.0):
    """The page's joint move today (each joint its own trapezoid, finishing on its own).

    q0, q1 [..., 3]; t [..., k] seconds; override a number or [...]. Returns q, qd, qdd [..., k, 3].
    """
    o = torch.as_tensor(override, dtype=q0.dtype, device=q0.device)
    o = o[..., None, None] if o.dim() else o
    v, acc = VMAX.to(q0.device) * o, AMAX.to(q0.device) * o
    dlt = (q1 - q0)[..., None, :]
    d, sgn = dlt.abs(), torch.sign(dlt)
    cruise = d >= v * v / acc
    ta = torch.where(cruise, v / acc, torch.sqrt(d / acc))           # time spent accelerating
    vp = acc * ta                                                     # peak speed
    tt = torch.where(cruise, d / v + v / acc, 2 * ta)                 # this joint's total time
    t = torch.minimum(t[..., None].clamp(min=0), tt)
    q0 = q0[..., None, :]
    tc = tt - ta                                                      # start of deceleration
    s = torch.where(t < ta, 0.5 * acc * t ** 2,
                    torch.where(t < tc, 0.5 * acc * ta ** 2 + vp * (t - ta),
                                d - 0.5 * acc * (tt - t) ** 2))
    sd = torch.where(t < ta, acc * t, torch.where(t < tc, vp, acc * (tt - t)))
    sdd = torch.where(t < ta, acc, torch.where(t < tc, torch.zeros_like(t), -acc))
    sdd = torch.where(t >= tt, torch.zeros_like(t), sdd)
    return q0 + sgn * s, sgn * sd, sgn * sdd


def sample_poses(n, device="cpu", generator=None):
    """Random joint configurations inside the limits (less the margin) whose elbow, wrist and tool clear the floor."""
    lim = LIM.to(device) + torch.tensor([JOINT_MARGIN, -JOINT_MARGIN], device=device)
    out = torch.empty(0, 3, device=device)
    while len(out) < n:
        q = lim[:, 0] + torch.rand(2 * n, 3, device=device, generator=generator) * (lim[:, 1] - lim[:, 0])
        out = torch.cat([out, q[(points_z(q) >= FLOOR_CLEAR).all(-1)]])
    return out[:n]
