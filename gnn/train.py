"""Train the IK graph network on the GPU.

Training data is generated on the fly: random joint configurations inside the
limits, pushed through forward kinematics to get the target each one reaches.

    python train.py                 # defaults: 30k steps, batch 4096
    python train.py --steps 5000    # quick run
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from arm import fk, goal_features, sample, wrap, Q_HALF
from model import IKGNN

HERE = Path(__file__).parent


def losses(model, q, tcp, phi, elbow, fk_weight=1.0):
    pred = model(goal_features(tcp, phi, elbow))
    q_half = Q_HALF.to(q.device)
    angle = (((pred - q) / q_half) ** 2).mean()
    p_tcp, p_phi = fk(pred)
    pos_mm = (p_tcp - tcp).norm(dim=-1)
    tool = wrap(p_phi - phi)
    # Joint-angle regression plus task-space error through differentiable FK (scaled to ~10 mm, ~1 deg).
    fk_terms = ((pos_mm / 10.0) ** 2).mean() * 0.01 + ((tool / (math.pi / 180)) ** 2).mean() * 0.001
    loss = angle + fk_weight * fk_terms
    return loss, pos_mm, tool.abs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--mp-steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    # CPU by default: the first CUDA run on this laptop's GTX 1050 coincided with a full system freeze.
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name()})" if dev == "cuda" else ""))

    model = IKGNN(args.hidden, args.mp_steps).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    g = torch.Generator(device=dev).manual_seed(1234)
    val = sample(50_000, dev, g)
    best_mm = float("inf")
    ckpt = HERE / "ik_gnn.pt"
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        # Learn the angles first, then ramp in the FK terms: at the start their errors are
        # hundreds of mm, and that gradient saturates the output and traps it in wrong poses.
        fk_weight = min(1.0, step / (0.3 * args.steps))
        loss, _, _ = losses(model, *sample(args.batch, dev), fk_weight=fk_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vloss, pos, tool = losses(model, *val)
            mean_mm, p95_mm = pos.mean().item(), pos.quantile(0.95).item()
            tool_deg = tool.mean().item() / (math.pi / 180)
            print(f"step {step:6d}  loss {loss.item():.5f}  val {vloss.item():.5f}  "
                  f"tcp err mean {mean_mm:6.2f} mm  p95 {p95_mm:6.2f} mm  "
                  f"tool {tool_deg:5.2f}°  [{time.time() - t0:5.0f}s]", flush=True)
            history.append({"step": step, "loss": loss.item(), "val_loss": vloss.item(),
                             "tcp_mean_mm": mean_mm, "tcp_p95_mm": p95_mm, "tool_deg": tool_deg})
            # Keep the checkpoint with the best gripper accuracy, not the lowest combined loss.
            if mean_mm < best_mm:
                best_mm = mean_mm
                torch.save({"state": model.state_dict(), "hidden": args.hidden, "steps": args.mp_steps}, ckpt)
    (HERE / "train_history.json").write_text(json.dumps(history))
    print(f"saved best checkpoint to {ckpt} (tcp err mean {best_mm:.2f} mm)")


if __name__ == "__main__":
    main()
