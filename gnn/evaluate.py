"""Compare the trained GNN with the closed-form IK on unseen targets.

Reports task-space error (where the gripper actually ends up after running the
predicted angles through forward kinematics) and joint-angle error, on random
targets and on the exact waypoints of the web page's pick-and-place program.
"""
import json
import math
from pathlib import Path

import torch

from arm import D, LIM, analytic_ik, fk, goal_features, sample, wrap
from model import IKGNN

HERE = Path(__file__).parent


def load(dev):
    ck = torch.load(HERE / "ik_gnn.pt", map_location=dev)
    model = IKGNN(ck["hidden"], ck["steps"]).to(dev)
    model.load_state_dict(ck["state"])
    return model.eval()


def metrics(model, tcp, phi, elbow, q_true=None):
    with torch.no_grad():
        lim = LIM.to(tcp.device)
        pred = torch.max(torch.min(model(goal_features(tcp, phi, elbow)), lim[:, 1]), lim[:, 0])
        p_tcp, p_phi = fk(pred)
        pos = (p_tcp - tcp).norm(dim=-1)
        tool = wrap(p_phi - phi).abs() / D
        out = {
            "tcp_err_mm": {"mean": pos.mean().item(), "median": pos.median().item(),
                           "p95": pos.quantile(0.95).item(), "p99": pos.quantile(0.99).item(), "max": pos.max().item()},
            "tool_err_deg": {"mean": tool.mean().item(), "p95": tool.quantile(0.95).item(), "max": tool.max().item()},
            "within_2mm_pct": (pos < 2).float().mean().item() * 100,
            "within_5mm_pct": (pos < 5).float().mean().item() * 100,
        }
        if q_true is not None:
            j = (pred - q_true).abs() / D
            out["joint_err_deg_mean"] = j.mean(0).tolist()
        return out, pred


def main():
    dev = "cpu"   # evaluation is quick; stay off the GPU (see train.py --device)
    model = load(dev)
    g = torch.Generator(device=dev).manual_seed(999)
    q, tcp, phi, elbow = sample(200_000, dev, g)

    # Sanity check: the closed-form solver must be exact on these targets.
    a_tcp, _ = fk(analytic_ik(tcp, phi, elbow))
    analytic_err = (a_tcp - tcp).norm(dim=-1).max().item()

    rand, _ = metrics(model, tcp, phi, elbow, q)

    # Waypoints used by the pick-and-place program on the page (tool pointing down, elbow up).
    pts = [(x, z) for x in (300.0, 540.0) for z in (30.0, 90.0, 150.0, 92.0, 152.0, 270.0)]
    w_tcp = torch.tensor(pts, device=dev)
    w_phi = torch.full((len(pts),), -90 * D, device=dev)
    w_elb = torch.full((len(pts),), -1.0, device=dev)
    prog, pred = metrics(model, w_tcp, w_phi, w_elb, analytic_ik(w_tcp, w_phi, w_elb))
    p_tcp, _ = fk(pred)
    prog["per_waypoint"] = [{"x": x, "z": z, "err_mm": round((p_tcp[i] - w_tcp[i]).norm().item(), 2)}
                            for i, (x, z) in enumerate(pts)]

    report = {"analytic_max_err_mm": analytic_err, "random_targets": rand, "program_waypoints": prog}
    (HERE / "eval.json").write_text(json.dumps(report, indent=2))

    print(f"closed-form IK sanity check: max error {analytic_err:.2e} mm")
    print("\nRandom reachable targets (200k, unseen):")
    r = rand["tcp_err_mm"]
    print(f"  TCP error  mean {r['mean']:.2f}  median {r['median']:.2f}  p95 {r['p95']:.2f}  p99 {r['p99']:.2f}  max {r['max']:.1f} mm")
    print(f"  tool angle mean {rand['tool_err_deg']['mean']:.2f}°  p95 {rand['tool_err_deg']['p95']:.2f}°")
    print(f"  within 2 mm: {rand['within_2mm_pct']:.1f}%   within 5 mm: {rand['within_5mm_pct']:.1f}%")
    print(f"  joint error mean (A1, A2, A3): " + ", ".join(f"{v:.2f}°" for v in rand["joint_err_deg_mean"]))
    print("\nPick-and-place waypoints:")
    for w in prog["per_waypoint"]:
        print(f"  X{w['x']:.0f} Z{w['z']:.0f}: {w['err_mm']:.2f} mm")
    print(f"\nwrote {HERE / 'eval.json'}")


if __name__ == "__main__":
    main()
