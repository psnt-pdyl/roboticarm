"""Train the motion PINN on physics alone: no example trajectories.

Each step draws random moves (start and end pose, duration, payload) and random times
within them. The network's motion goes through the equations of motion to get the
motor torques, and the loss is what physics says about those torques and poses:

    effort      mean over the move of sum_i (tau_i / tau_max_i)^2   (minimise)
    torque      motors past their limits                            (penalty)
    joints      joints past their limits                            (penalty)
    floor       elbow, wrist or tool point below 20 mm              (penalty)

Durations cover the page's joint-move time at 50-200 % speed override.

    python train.py --device cuda          # defaults: 20000 steps, 512 moves x 16 times
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from dynamics import D, FLOOR_CLEAR, JOINT_MARGIN, LIM, TAU_MAX, blend, inverse_dynamics, move_time, points_z, sample_poses
from model import Q_HALF, MotionPINN

HERE = Path(__file__).parent
PENALTY = 1000.0          # motor torque limits and floor clearance
JOINT_PENALTY = 1e5       # joint limits are hard stops: much stiffer, kept 1 degree inside
T_MIN = 0.05
MAX_TERMS = ("peak", "over_deg", "floor_mm")


def sample_moves(n, device, generator=None):
    """Start and end poses, duration (s) and payload (kg) for n random moves."""
    r = lambda *s: torch.rand(*s, device=device, generator=generator)
    q0, q1 = sample_poses(n, device, generator), sample_poses(n, device, generator)
    override = torch.exp(math.log(0.5) + r(n) * math.log(4.0))
    T = (move_time(q0, q1) / override).clamp(min=T_MIN)
    payload = torch.where(r(n) < 0.5, torch.zeros(n, device=device), r(n))
    return q0, q1, T, payload


def physics_terms(q, qd, qdd, payload):
    """Per-sample effort and constraint violations [..., ] for motions [..., 3]."""
    dev = q.device
    r = inverse_dynamics(q, qd, qdd, payload) / TAU_MAX.to(dev)
    lim, half = LIM.to(dev), Q_HALF.to(dev)
    lo, hi = lim[:, 0] + JOINT_MARGIN, lim[:, 1] - JOINT_MARGIN
    joints = (torch.relu((q - hi) / half) ** 2 + torch.relu((lo - q) / half) ** 2).sum(-1)
    z = points_z(q)
    floor = (torch.relu((FLOOR_CLEAR - z) / 100) ** 2).sum(-1)
    over = (torch.relu(q - lim[:, 1]) + torch.relu(lim[:, 0] - q)).max(-1).values / D   # past the true limits, deg
    return {"effort": (r ** 2).sum(-1), "torque": (torch.relu(r.abs() - 1) ** 2).sum(-1),
            "joints": joints, "floor": floor, "peak": r.abs().max(-1).values,
            "over_deg": over, "floor_mm": torch.relu(-z).max(-1).values}


def penalty(terms):
    return PENALTY * (terms["torque"] + terms["floor"]) + JOINT_PENALTY * terms["joints"]


def loss_of(terms):
    return (terms["effort"] + penalty(terms)).mean()


def evaluate(model, moves, k=64, chunk=250):
    """Physics of the network's motion on uniform times, per move: dict of [n] tensors."""
    parts = []
    for i in range(0, len(moves[0]), chunk):
        q0, q1, T, p = (x[i:i + chunk] for x in moves)
        s = torch.linspace(0, 1, k, device=q0.device).expand(len(q0), k)
        ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
        q, qd, qdd = model.motion(ex(q0), ex(q1), s, ex(T), ex(p))
        parts.append(reduce(physics_terms(q.detach(), qd.detach(), qdd.detach(), ex(p))))
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def evaluate_blend(moves, k=64):
    """The same measures for the page's current joint blend over the same durations."""
    q0, q1, T, p = moves
    override = (move_time(q0, q1) / T).clamp(min=0.05)   # the speed override that makes the blend take T
    t = torch.linspace(0, 1, k, device=q0.device) * T[:, None]
    q, qd, qdd = blend(q0, q1, t, override)
    return reduce(physics_terms(q, qd, qdd, p[:, None].expand(-1, k)))


def reduce(terms):
    """Average each term over the move's times; peaks and worst violations as the maximum."""
    return {k: (v.max(-1).values if k in MAX_TERMS else v.mean(-1)) for k, v in terms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--times", type=int, default=16, help="collocation times per move")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    model = MotionPINN(args.hidden, args.layers).to(dev)
    print(f"device: {dev}  parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    val = sample_moves(4000, dev, torch.Generator(device=dev).manual_seed(1234))
    ref = evaluate_blend(val)
    print(f"joint blend on the validation moves: effort {ref['effort'].mean():.4f}  "
          f"peak torque {ref['peak'].mean() * 100:.0f} % of limit  over limit {(ref['peak'] > 1).float().mean() * 100:.1f} %")
    best, history, ckpt, t0 = float("inf"), [], HERE / "pinn.pt", time.time()
    for step in range(1, args.steps + 1):
        model.train()
        q0, q1, T, p = sample_moves(args.batch, dev)
        k = args.times
        s = torch.rand(args.batch, k, device=dev)
        ex = lambda x: x[:, None].expand(-1, k, *x.shape[1:])
        q, qd, qdd = model.motion(ex(q0), ex(q1), s, ex(T), ex(p))
        loss = loss_of(physics_terms(q, qd, qdd, ex(p)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == args.steps:
            model.eval()
            v = evaluate(model, val)
            vloss = (v["effort"] + penalty(v)).mean().item()
            row = {"step": step, "loss": loss.item(), "val_loss": vloss, "effort": v["effort"].mean().item(),
                   "peak_pct": v["peak"].mean().item() * 100, "over_limit_pct": (v["peak"] > 1).float().mean().item() * 100,
                   "past_joint_limit_pct": (v["over_deg"] > 0).float().mean().item() * 100,
                   "max_joint_overshoot_deg": v["over_deg"].max().item()}
            history.append(row)
            print(f"step {step:6d}  loss {row['loss']:.4f}  val {vloss:.4f}  effort {row['effort']:.4f} "
                  f"(blend {ref['effort'].mean():.4f})  peak {row['peak_pct']:.0f} %  over torque limit {row['over_limit_pct']:.1f} %  "
                  f"past joint limit {row['past_joint_limit_pct']:.1f} % (max {row['max_joint_overshoot_deg']:.2f}°)  "
                  f"[{time.time() - t0:4.0f}s]", flush=True)
            if vloss < best:
                best = vloss
                torch.save({"state": model.state_dict(), "hidden": args.hidden, "layers": args.layers}, ckpt)
    (HERE / "train_history.json").write_text(json.dumps(history))
    print(f"saved best checkpoint to {ckpt} (val loss {best:.4f})")


if __name__ == "__main__":
    main()
