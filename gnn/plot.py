"""Plot the training curves from train_history.json (written by train.py) to training_curves.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def panel(ax, steps, series, title, fmt, warmup, labelled=None):
    for i, (label, ys, color) in enumerate(series):
        ax.plot(steps, ys, color=color, lw=2, label=label, solid_capstyle="round")
        if labelled is None or i in labelled:
            ax.annotate(fmt(ys[-1]), (steps[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", which="major", color=GRID, lw=0.8)
    ax.axvline(warmup, color=AXIS, lw=1, ls="--", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")


def main():
    hist = json.loads((HERE / "train_history.json").read_text())
    steps = [h["step"] for h in hist]
    warmup = 0.3 * steps[-1]   # train.py ramps the FK loss terms in over the first 30% of steps

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True, facecolor=SURFACE)
    panel(axes[0], steps, [("mean", [h["tcp_mean_mm"] for h in hist], BLUE),
                           ("95th percentile", [h["tcp_p95_mm"] for h in hist], ORANGE)],
          "Gripper position error (mm, log scale)", lambda v: f"{v:.2f} mm", warmup)
    panel(axes[1], steps, [("mean", [h["tool_deg"] for h in hist], BLUE)],
          "Tool angle error (degrees, log scale)", lambda v: f"{v:.2f}°", warmup)
    # Train and validation loss converge, so only label the validation end to avoid overlap.
    panel(axes[2], steps, [("train", [h["loss"] for h in hist], BLUE),
                           ("validation", [h["val_loss"] for h in hist], ORANGE)],
          "Loss (log scale)", lambda v: f"{v:.0e}", warmup, labelled={1})
    axes[0].annotate("FK terms at full weight", (warmup, 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -2), textcoords="offset points", va="top", color=MUTED, fontsize=8)
    axes[2].set_xlabel("training step", color=MUTED, fontsize=9)
    fig.suptitle("IK graph network training", x=0.07, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.86, top=0.92, bottom=0.07, hspace=0.35)
    out = HERE / "training_curves.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
