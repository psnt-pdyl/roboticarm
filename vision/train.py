"""Train the part-finding network on camera scenes rendered on the fly.

    python train.py --device cuda          # defaults: 6000 steps, batch 128
"""
import argparse
import time
from pathlib import Path

import torch

from model import VisionNet
from scene import sample

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    model = VisionNet(args.width).to(dev)
    print(f"device: {dev}  parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    g = torch.Generator(device=dev).manual_seed(1234)
    val_img, val_xz = sample(2000, dev, g)
    best = float("inf")
    ckpt = HERE / "vision_net.pt"
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        img, xz = sample(args.batch, dev)
        # Huber, not squared error: a scene where the arm hides the part can be missed by
        # hundreds of mm, and a squared loss lets that one sample derail the whole update.
        loss = torch.nn.functional.smooth_l1_loss(model(img) / 10.0, xz / 10.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                err = (model(val_img) - val_xz).norm(dim=-1)
            mean_mm = err.mean().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  val error mean {mean_mm:6.2f} mm  "
                  f"p95 {err.quantile(0.95).item():6.2f} mm  max {err.max().item():6.1f} mm  "
                  f"[{time.time() - t0:4.0f}s]", flush=True)
            if mean_mm < best:
                best = mean_mm
                torch.save({"state": model.state_dict(), "width": args.width}, ckpt)
    print(f"saved best checkpoint to {ckpt} (val error mean {best:.2f} mm)")


if __name__ == "__main__":
    main()
