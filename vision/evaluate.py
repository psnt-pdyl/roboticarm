"""Measure how precisely the trained network finds the part, on unseen scenes.

Reports the position error in mm, with and without the arm in view, and the share of
scenes where the error is small enough for the gripper to catch the part (the page
grips a part whose centre is within 22 mm of the gripper tip).
"""
import json
from pathlib import Path

import torch

from model import VisionNet
from scene import sample

HERE = Path(__file__).parent
GRIP_TOLERANCE_MM = 22.0


def stats(err):
    return {"mean": err.mean().item(), "median": err.median().item(), "p95": err.quantile(0.95).item(),
            "p99": err.quantile(0.99).item(), "max": err.max().item(),
            "graspable_pct": (err < GRIP_TOLERANCE_MM).float().mean().item() * 100}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(HERE / "vision_net.pt", map_location=dev)
    model = VisionNet(ck["width"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    g = torch.Generator(device=dev).manual_seed(999)
    report = {}
    for name, augment in (("clean", False), ("noisy", True)):
        errs = []
        with torch.no_grad():
            for _ in range(10):
                img, xz = sample(1000, dev, g, augment=augment)
                errs.append((model(img) - xz).norm(dim=-1))
        report[name] = stats(torch.cat(errs))

    (HERE / "eval.json").write_text(json.dumps(report, indent=2))
    for name, s in report.items():
        print(f"{name:5s} scenes (10k): error mean {s['mean']:.2f}  median {s['median']:.2f}  p95 {s['p95']:.2f}  "
              f"p99 {s['p99']:.2f}  max {s['max']:.1f} mm   graspable {s['graspable_pct']:.2f}%")
    print(f"wrote {HERE / 'eval.json'}")


if __name__ == "__main__":
    main()
