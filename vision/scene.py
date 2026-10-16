"""The work cell's virtual camera: what it sees and how a scene is rendered.

The camera looks side-on at the strip of floor where parts appear. index.html renders
the same image from the same constants (export.py ships them with the weights), so the
network sees in the browser exactly what it was trained on.

Units: millimetres, x along the floor, z up. Renders run batched on any torch device.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Appended, not prepended: gnn/ has its own model.py, which must not shadow this folder's.
sys.path.append(str(Path(__file__).resolve().parent.parent / "gnn"))
from arm import L1, L2, L3, LIM, SH, analytic_ik  # noqa: E402

# View window and image size: 3.5 mm per pixel on both axes.
X0, X1, Z0, Z1 = 100.0, 660.0, -20.0, 120.0
W, H = 160, 40
SS = 4                                    # SS x SS samples per pixel (anti-aliasing)
BS = 60.0                                 # part edge length
PART_X = (140.0, 630.0)                   # range of part centres seen in training

COLORS = {"bg": "#dfe3e7", "floor": "#4b5560", "arm": "#efb300", "dark": "#28323d", "steel": "#a7b2bd"}
# The arm as the camera models it: round-capped strokes (radius in mm), drawn in this order.
RADII = {"link1": 31, "shoulder": 44, "link2": 22, "elbow": 31, "rod": 13, "plate": 10, "finger": 6, "wrist": 21}


def rgb(hex_color):
    return torch.tensor([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def hsv_to_rgb(h, s, v):
    k = (torch.tensor([5.0, 3.0, 1.0], device=h.device) + h[:, None] * 6) % 6
    return v[:, None] - v[:, None] * s[:, None] * torch.minimum(k, 4 - k).clamp(0, 1)


def arm_strokes(q, g):
    """The arm's strokes for joint angles q [B,3] and gripper opening g [B], in drawing order.

    Returns (start [B,2], end [B,2], radius, colour key) tuples; a joint cap is a stroke
    whose start and end coincide. Mirrors the shapes of the main view in index.html.
    """
    a1 = q[:, 0]
    a2 = a1 + q[:, 1]
    a3 = a2 + q[:, 2]
    s = torch.tensor(SH, device=q.device).expand(len(q), 2)
    e = s + L1 * torch.stack([a1.cos(), a1.sin()], -1)
    w = e + L2 * torch.stack([a2.cos(), a2.sin()], -1)
    u = torch.stack([a3.cos(), a3.sin()], -1)
    n = torch.stack([-u[:, 1], u[:, 0]], -1)
    hw = (12 + 43 * g)[:, None]           # finger inner half-gap, as halfW() in index.html
    strokes = [(s, e, RADII["link1"], "arm"), (s, s, RADII["shoulder"], "dark"),
               (e, w, RADII["link2"], "arm"), (e, e, RADII["elbow"], "dark"),
               (w, w + u * (L3 - 68), RADII["rod"], "dark"),
               (w + u * (L3 - 62) - n * 64, w + u * (L3 - 62) + n * 64, RADII["plate"], "dark")]
    for sgn in (-1, 1):
        off = n * sgn * (hw + 6)
        strokes.append((w + u * (L3 - 50) + off, w + u * (L3 + 12) + off, RADII["finger"], "steel"))
    strokes.append((w, w, RADII["wrist"], "dark"))
    return strokes


def _within(px, pz, a, b, r):
    """Points within r of segment a-b (a, b: [B,2]) -> mask [B, rows, cols]."""
    ax, az = a[:, 0, None, None], a[:, 1, None, None]
    dx, dz = (b - a)[:, 0, None, None], (b - a)[:, 1, None, None]
    t = (((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz).clamp_min(1e-9)).clamp(0, 1)
    return (px - ax - t * dx) ** 2 + (pz - az - t * dz) ** 2 <= r * r


def render(part_xz, part_rgb, q, g, show_arm):
    """Camera images [B,3,H,W] in [0,1].

    part_xz [B,2] part centre (mm), part_rgb [B,3], q [B,3] and g [B] arm pose and
    gripper opening, show_arm [B] bool (False: the arm is out of view).
    """
    dev, B = part_xz.device, len(part_xz)
    px = (X0 + (torch.arange(W * SS, device=dev) + 0.5) * (X1 - X0) / (W * SS)).view(1, 1, -1)
    pz = (Z1 - (torch.arange(H * SS, device=dev) + 0.5) * (Z1 - Z0) / (H * SS)).view(1, -1, 1)
    col = {k: rgb(v).to(dev).view(1, 3, 1, 1) for k, v in COLORS.items()}
    img = col["bg"].repeat(B, 1, H * SS, W * SS)
    img = torch.where((pz < 0)[:, None], col["floor"], img)
    part = ((px - part_xz[:, 0, None, None]).abs() <= BS / 2) & ((pz - part_xz[:, 1, None, None]).abs() <= BS / 2)
    img = torch.where(part[:, None], part_rgb.view(B, 3, 1, 1), img)
    for a, b, r, key in arm_strokes(q, g):
        img = torch.where((_within(px, pz, a, b, r) & show_arm[:, None, None])[:, None], col[key], img)
    return F.avg_pool2d(img, SS)


def sample(n, device="cpu", generator=None, augment=True):
    """Random scenes and the part centre in each: images [n,3,H,W], xz [n,2] (mm).

    One part on the floor in a random colour. The arm is posed with its gripper near the
    view in 40% of scenes, in a random pose in 30% (often out of view), and absent in 30%.
    With augment, lighting, colour balance and sensor noise vary too.
    """
    def r(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    x = PART_X[0] + r(n) * (PART_X[1] - PART_X[0])
    xz = torch.stack([x, torch.full_like(x, BS / 2)], -1)
    part_rgb = hsv_to_rgb(r(n), 0.25 + 0.75 * r(n), 0.25 + 0.7 * r(n))

    kind = r(n)
    target = torch.stack([X0 + r(n) * (X1 - X0), Z0 + 40 + r(n) * 200], -1)
    phi = (-150 + 120 * r(n)) * math.pi / 180
    lim = LIM.to(device)
    q = torch.where((kind < 0.4)[:, None], analytic_ik(target, phi, -torch.ones(n, device=device)),
                    lim[:, 0] + r(n, 3) * (lim[:, 1] - lim[:, 0]))
    q = torch.max(torch.min(q, lim[:, 1]), lim[:, 0])
    img = render(xz, part_rgb, q, r(n), kind < 0.7)

    if augment:
        gain = (0.75 + 0.5 * r(n, 1, 1, 1)) * (0.9 + 0.2 * r(n, 3, 1, 1))
        img = img * gain + (r(n, 1, 1, 1) - 0.5) * 0.1
        img = img + torch.randn(img.shape, device=device, generator=generator) * 0.04 * r(n, 1, 1, 1)
        img = img.clamp(0, 1)
    return img, xz
