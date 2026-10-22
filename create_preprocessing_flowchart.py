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


def box(ax, x, y, w, h, number, title, details, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8, edgecolor=color, facecolor=PANEL,
        transform=ax.transAxes,
    ))
    ax.text(x + 0.025, y + h - 0.04, number, transform=ax.transAxes,
            color=color, fontsize=15, fontweight="bold", va="top")
    ax.text(x + 0.09, y + h - 0.04, title, transform=ax.transAxes,
            color=TEXT, fontsize=13, fontweight="bold", va="top")
    ax.text(x + 0.09, y + h - 0.09, details, transform=ax.transAxes,
            color=MUTED, fontsize=10, va="top", linespacing=1.55)


def arrow(ax, x1, x2):
    ax.add_patch(FancyArrowPatch(
        (x1, 0.51), (x2, 0.51), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=18, linewidth=2,
        color=TEAL,
    ))


def main():
    fig = plt.figure(figsize=(16, 8.5), dpi=180, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()

    ax.text(0.06, 0.91, "SLEEP EEG PREPROCESSING FLOW", transform=ax.transAxes,
            color=TEAL, fontsize=13, fontweight="bold")
    ax.text(0.06, 0.85, "From preprocessed epoch to model-ready tensor", transform=ax.transAxes,
            color=TEXT, fontsize=25, fontweight="bold")
    ax.text(0.06, 0.80, "The same operations are applied before every sleep-stage prediction.",
            transform=ax.transAxes, color=MUTED, fontsize=11)

    y, w, h = 0.38, 0.16, 0.25
    positions = [0.06, 0.25, 0.44, 0.63, 0.82]
    box(ax, positions[0], y, w, h, "01", "LOAD EPOCH", "NPZ subject file\nInput: x [epochs, time, channels]\nTypical epoch: 30 s", TEAL)
    box(ax, positions[1], y, w, h, "02", "VALIDATE", "Check x and y exist\nCheck 3D signal shape\nCheck epoch/label counts", AMBER)
    box(ax, positions[2], y, w, h, "03", "SELECT CHANNELS", "Keep EEG channels\nOptionally remove EOG\nEEG-only: first 6 channels", BLUE)
    box(ax, positions[3], y, w, h, "04", "NORMALIZE", "Use training mean/std\nPer-channel z-score:\n(x - mean) / std", GREEN)
    box(ax, positions[4], y, w, h, "05", "TRANSPOSE", "Reorder dimensions\n[epochs, time, channels]\n-> [epochs, channels, time]", TEAL)

    arrow(ax, 0.222, 0.245)
    arrow(ax, 0.412, 0.435)
    arrow(ax, 0.602, 0.625)
    arrow(ax, 0.792, 0.815)

    ax.text(0.06, 0.29, "MODEL-READY OUTPUT", transform=ax.transAxes,
            color=TEAL, fontsize=10, fontweight="bold")
    ax.text(0.06, 0.245, "A normalized epoch is converted to a PyTorch tensor and passed to Conv1d as [channels, time].",
            transform=ax.transAxes, color=TEXT, fontsize=12)

    ax.add_patch(FancyBboxPatch(
        (0.06, 0.08), 0.88, 0.10,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8, edgecolor="#31516a", facecolor="#142b40",
        transform=ax.transAxes,
    ))
    ax.text(0.08, 0.13,
            "raw x  [epochs, time, channels]   ->   selected + normalized   ->   tensor [channels, time]",
            transform=ax.transAxes, color=TEXT, fontsize=12, family="monospace", va="center")

    fig.savefig(OUTPUT, dpi=180, facecolor=BG)
    print(f"Saved {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
