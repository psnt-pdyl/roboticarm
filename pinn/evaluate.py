"""Compare the trained motion PINN with today's joint blend and with a per-move optimum.

The per-move optimum is the same physics loss minimised separately for each move, over
24 free polynomial coefficients, by thousands of Adam steps. It is the best motion this
path shape can give, so the gap to it measures how much the single trained network
gives up in exchange for answering any move in one forward pass.

Writes eval.json and example_move.png.
"""
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dynamics import TAU_MAX, PART_MASS, PAYLOAD_MAX, blend, inverse_dynamics, move_time
from model import MotionPINN, compose
from train import evaluate, evaluate_blend, penalty, physics_terms, reduce, sample_moves

sys.path.append(str(Path(__file__).resolve().parent.parent / "gnn"))
from arm import analytic_ik  # noqa: E402

HERE = Path(__file__).parent
K = 8                 # polynomial coefficients per joint for the per-move optimum
D = math.pi / 180


def legendre(x, k):
    P = [torch.ones_like(x), x]
    for n in range(1, k - 1):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return torch.stack(P[:k], -1)


def optimum(moves, steps=3000, k=64):
    """Per-move optimised motion: returns (terms per move, coefficients)."""
    q0, q1, T, p = moves
    n, dev = len(q0), q0.device
    c = torch.zeros(n, 3, K, device=dev, requires_grad=True)
    opt = torch.optim.Adam([c], lr=0.02)
    s = torch.linspace(0, 1, k, device=dev).expand(n, k)
    ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
    shape = lambda s_: torch.einsum("nkb,njb->nkj", legendre(2 * s_ - 1, K), c)
    for i in range(steps):
        q, qd, qdd = compose(shape, ex(q0), ex(q1), s, ex(T))
        t = physics_terms(q, qd, qdd, ex(p))
        loss = (t["effort"] + penalty(t)).mean(-1).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == steps // 2:
            for g in opt.param_groups:
                g["lr"] = 0.005
    q, qd, qdd = compose(shape, ex(q0), ex(q1), s, ex(T))
    return reduce(physics_terms(q.detach(), qd.detach(), qdd.detach(), ex(p))), c.detach()


def summary(t, ref=None):
    out = {"effort": t["effort"].mean().item(), "peak_torque_pct": t["peak"].mean().item() * 100,
           "max_peak_torque_pct": t["peak"].max().item() * 100,
           "moves_over_torque_limit_pct": (t["peak"] > 1.0005).float().mean().item() * 100,
           "moves_past_joint_limit_pct": (t["over_deg"] > 0.01).float().mean().item() * 100,
           "max_joint_overshoot_deg": t["over_deg"].max().item(),
           "moves_into_floor_pct": (t["floor_mm"] > 0.1).float().mean().item() * 100,
           "max_floor_penetration_mm": t["floor_mm"].max().item()}
    if ref is not None:
        out["effort_vs_blend_pct"] = (t["effort"].mean() / ref["effort"].mean() - 1).item() * 100
    return out


def page_moves(dev):
    """The joint moves the page actually makes, at 100 % and 200 % speed override."""
    def pose(x, z, elbow):
        return analytic_ik(torch.tensor([[x, z]], device=dev), torch.tensor([-90 * D], device=dev),
                           torch.tensor([elbow], device=dev))[0]
    defs = [("park -> above part (vision)", pose(-185, 300, 1.0), pose(313, 270, -1.0), 0.0),
            ("above part -> bin, carrying a part", pose(313, 270, -1.0), pose(-185, 300, 1.0), PART_MASS),
            ("above A -> above B, carrying a part", pose(300, 270, -1.0), pose(540, 270, -1.0), PART_MASS)]
    names, q0, q1, T, p = [], [], [], [], []
    for override in (1.0, 2.0):
        for name, a, b, m in defs:
            names.append(f"{name} @ {override * 100:.0f} %")
            q0.append(a), q1.append(b), p.append(m)
            T.append(move_time(a, b) / override)
    st = lambda v: torch.stack(v) if torch.is_tensor(v[0]) else torch.tensor(v, device=dev)
    return names, (st(q0), st(q1), st(T), st(p))


def plot_move(model, move, c, path):
    """Joint angles and motor torques over one move: blend, PINN and per-move optimum."""
    q0, q1, T, p = (x[None] for x in move)
    k = 400
    s = torch.linspace(0, 1, k, device=q0.device)[None]
    ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
    t = (s * T[:, None])[0].cpu()
    curves = {}
    qb, qdb, qddb = blend(q0, q1, s * T[:, None], move_time(q0, q1) / T)
    curves["joint blend (today)"] = (qb[0], inverse_dynamics(qb, qdb, qddb, ex(p))[0])
    qn, qdn, qddn = model.motion(ex(q0), ex(q1), s, ex(T), ex(p))
    curves["PINN"] = (qn[0].detach(), inverse_dynamics(qn, qdn, qddn, ex(p))[0].detach())
    shape = lambda s_: torch.einsum("nkb,njb->nkj", legendre(2 * s_ - 1, K), c[None])
    qo, qdo, qddo = compose(shape, ex(q0), ex(q1), s, ex(T))
    curves["per-move optimum"] = (qo[0].detach(), inverse_dynamics(qo, qdo, qddo, ex(p))[0].detach())

    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    style = {"joint blend (today)": ("#eb6834", "-"), "PINN": ("#2a78d6", "-"), "per-move optimum": ("#1baf7a", "--")}
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True, facecolor=SURFACE)
    names = ["A1 shoulder", "A2 elbow", "A3 wrist"]
    for j in range(3):
        for name, (q, tau) in curves.items():
            col, ls = style[name]
            axes[j, 0].plot(t, (q[:, j] / D).cpu(), color=col, ls=ls, lw=2, label=name)
            axes[j, 1].plot(t, (tau[:, j] / TAU_MAX[j] * 100).cpu(), color=col, ls=ls, lw=2, label=name)
        axes[j, 0].set_title(f"{names[j]}: angle (°)", loc="left", color=INK, fontsize=10, fontweight="bold")
        axes[j, 1].set_title(f"{names[j]}: torque (% of {TAU_MAX[j]:.0f} N·m limit)", loc="left", color=INK,
                             fontsize=10, fontweight="bold")
        for ax in axes[j]:
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
        axes[j, 1].axhline(0, color=AXIS, lw=1)
    axes[0, 1].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
    for ax in axes[2]:
        ax.set_xlabel("time (s)", color=MUTED, fontsize=9)
    fig.suptitle("Swing from above the part to the bin, carrying a part (100 % speed)", x=0.06, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.08, hspace=0.45, wspace=0.18)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(HERE / "pinn.pt", map_location=dev)
    model = MotionPINN(ck["hidden"], ck["layers"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    report = {}
    moves = sample_moves(1000, dev, torch.Generator(device=dev).manual_seed(999))
    b = evaluate_blend(moves)
    t0 = time.time()
    n = evaluate(model, moves)
    if dev == "cuda":
        torch.cuda.synchronize()
    pinn_s = time.time() - t0
    t0 = time.time()
    o, _ = optimum(moves)
    opt_s = time.time() - t0
    report["random_moves"] = {"joint_blend": summary(b), "pinn": summary(n, b), "per_move_optimum": summary(o, b),
                              "pinn_gap_to_optimum_pct": (n["effort"].mean() / o["effort"].mean() - 1).item() * 100,
                              "seconds_for_1000_moves": {"pinn": pinn_s, "per_move_optimum": opt_s}}

    names, pm = page_moves(dev)
    pb, pn = evaluate_blend(pm), evaluate(model, pm)
    po, coeffs = optimum(pm)
    report["page_moves"] = [{"move": nm, "T_s": pm[2][i].item(),
                             **{f"{m}_{k}": v[k][i].item() * (100 if k == "peak" else 1)
                                for m, v in (("blend", pb), ("pinn", pn), ("optimum", po)) for k in ("effort", "peak")}}
                            for i, nm in enumerate(names)]
    (HERE / "eval.json").write_text(json.dumps(report, indent=2))

    r = report["random_moves"]
    print(f"1000 random unseen moves (50-200 % speed, 0-{PAYLOAD_MAX:g} kg payload):")
    print(f"  {'':18s}{'effort':>9s}{'vs blend':>10s}{'peak torque':>13s}{'over torque lim':>17s}"
          f"{'past joint limit':>24s}{'into the floor':>24s}")
    for key, label in (("joint_blend", "joint blend"), ("pinn", "PINN"), ("per_move_optimum", "per-move optimum")):
        s = r[key]
        vs = f"{s['effort_vs_blend_pct']:+.1f} %" if "effort_vs_blend_pct" in s else "-"
        print(f"  {label:18s}{s['effort']:9.4f}{vs:>10s}{s['peak_torque_pct']:11.0f} %{s['moves_over_torque_limit_pct']:15.1f} %"
              f"{s['moves_past_joint_limit_pct']:12.1f} % (max {s['max_joint_overshoot_deg']:5.2f}°)"
              f"{s['moves_into_floor_pct']:11.1f} % (max {s['max_floor_penetration_mm']:5.0f} mm)")
    print(f"  PINN effort is {r['pinn_gap_to_optimum_pct']:.1f} % above the per-move optimum; "
          f"time for all 1000 moves: PINN {pinn_s * 1000:.0f} ms, per-move optimisation {opt_s:.0f} s")
    print("\nThe page's own joint moves (effort, peak torque % of limit):")
    for row in report["page_moves"]:
        print(f"  {row['move']:44s} T {row['T_s']:.2f} s   blend {row['blend_effort']:.3f} / {row['blend_peak']:3.0f} %   "
              f"PINN {row['pinn_effort']:.3f} / {row['pinn_peak']:3.0f} %   optimum {row['optimum_effort']:.3f} / {row['optimum_peak']:3.0f} %")
    plot_move(model, tuple(x[1] for x in pm), coeffs[1], HERE / "example_move.png")
    print(f"\nwrote {HERE / 'eval.json'} and {HERE / 'example_move.png'}")


if __name__ == "__main__":
    main()
