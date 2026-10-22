from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT = Path("sleep_preprocessing_flowchart.png")
BG = "#08111f"
PANEL = "#122236"
TEXT = "#edf4f7"
MUTED = "#a7bac7"
TEAL = "#5eead4"
AMBER = "#fbbf24"
BLUE = "#60a5fa"
GREEN = "#86efac"


def step(ax, y, number, title, details, color):
    x, width, height = 0.16, 0.68, 0.115
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8, edgecolor=color, facecolor=PANEL,
        transform=ax.transAxes,
    ))
    ax.text(x + 0.025, y + height - 0.035, number, transform=ax.transAxes,
            color=color, fontsize=16, fontweight="bold", va="top")
    ax.text(x + 0.105, y + height - 0.035, title, transform=ax.transAxes,
            color=TEXT, fontsize=14, fontweight="bold", va="top")
    ax.text(x + 0.105, y + height - 0.072, details, transform=ax.transAxes,
            color=MUTED, fontsize=10.5, va="top", linespacing=1.45)


def down_arrow(ax, start_y, end_y):
    ax.add_patch(FancyArrowPatch(
        (0.5, start_y), (0.5, end_y), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=16, linewidth=2, color=TEAL,
    ))


def main():
    fig = plt.figure(figsize=(12, 16), dpi=180, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()

    ax.text(0.08, 0.95, "SLEEP EEG PREPROCESSING FLOW", transform=ax.transAxes,
            color=TEAL, fontsize=13, fontweight="bold")
    ax.text(0.08, 0.91, "From preprocessed epoch to model-ready tensor", transform=ax.transAxes,
            color=TEXT, fontsize=24, fontweight="bold")
    ax.text(0.08, 0.875, "The same operations are applied before every sleep-stage prediction.",
            transform=ax.transAxes, color=MUTED, fontsize=11)

    steps = [
        ("01", "LOAD EPOCH", "NPZ subject file\nInput: x [epochs, time, channels]\nTypical epoch: 30 seconds", TEAL),
        ("02", "VALIDATE", "Check x and y exist\nCheck 3D signal shape\nCheck epoch and label counts", AMBER),
        ("03", "SELECT CHANNELS", "Keep EEG channels\nOptionally remove EOG\nEEG-only mode keeps the first 6 channels", BLUE),
        ("04", "NORMALIZE", "Use training-set mean and standard deviation\nPer-channel z-score: (x - mean) / std", GREEN),
        ("05", "TRANSPOSE", "Reorder dimensions for Conv1d\n[epochs, time, channels] -> [epochs, channels, time]", TEAL),
    ]
    ys = [0.70, 0.565, 0.43, 0.295, 0.16]
    for index, (number, title, details, color) in enumerate(steps):
        step(ax, ys[index], number, title, details, color)
        if index < len(steps) - 1:
            down_arrow(ax, ys[index] - 0.006, ys[index + 1] + 0.121)

    ax.text(0.08, 0.105, "MODEL-READY OUTPUT", transform=ax.transAxes,
            color=TEAL, fontsize=10, fontweight="bold")
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.035), 0.84, 0.055,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8, edgecolor="#31516a", facecolor="#142b40",
        transform=ax.transAxes,
    ))
    ax.text(0.105, 0.062,
            "PyTorch tensor: [channels, time]  ->  Conv1d sleep-stage model",
            transform=ax.transAxes, color=TEXT, fontsize=12, family="monospace", va="center")

    fig.savefig(OUTPUT, dpi=180, facecolor=BG)
    print(f"Saved {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
