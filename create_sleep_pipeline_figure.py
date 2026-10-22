from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


OUTPUT = Path("sleep_signal_processing_pipeline.png")
BG = "#08111f"
PANEL = "#101d2d"
PANEL_2 = "#14263a"
TEXT = "#edf4f7"
MUTED = "#9db0bd"
TEAL = "#5eead4"
CORAL = "#fb7185"
AMBER = "#fbbf24"
BLUE = "#60a5fa"
GREEN = "#86efac"


def rounded_box(ax, x, y, width, height, title, body, accent):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2, edgecolor=accent, facecolor=PANEL,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + height - 0.065, title, transform=ax.transAxes,
            color=accent, fontsize=13, fontweight="bold", va="top")
    ax.text(x + 0.025, y + height - 0.115, body, transform=ax.transAxes,
            color=TEXT, fontsize=9.5, va="top", linespacing=1.5)


def arrow(ax, start, end, color=TEAL):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=16, linewidth=1.8, color=color,
        connectionstyle="arc3,rad=0.0",
    ))


def draw_signal(ax, x, y, width, height, seed, color, label):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 420)
    base = 0.08 * rng.normal(size=t.size)
    smooth = np.convolve(base, np.ones(9) / 9, mode="same")
    signal = 0.28 * np.sin(2 * np.pi * (4.5 + seed * 0.3) * t) + smooth
    if seed == 2:
        signal += 0.18 * np.sin(2 * np.pi * 13 * t)
    signal = signal / max(np.max(np.abs(signal)), 1e-6)
    px = x + t * width
    py = y + height * (0.5 + 0.34 * signal)
    ax.plot(px, py, transform=ax.transAxes, color=color, linewidth=1.2, clip_on=False)
    ax.text(x, y + height + 0.012, label, transform=ax.transAxes,
            color=MUTED, fontsize=8.5, va="bottom")


def probability_bars(ax, x, y, width, height):
    names = ["W", "N1", "N2", "N3", "R"]
    values = [0.08, 0.04, 0.63, 0.11, 0.14]
    colors = [GREEN, AMBER, "#facc15", BLUE, "#c084fc"]
    row_h = height / len(names)
    for i, (name, value, color) in enumerate(zip(names, values, colors)):
        yy = y + height - (i + 1) * row_h + row_h * 0.22
        ax.text(x, yy + row_h * 0.28, name, transform=ax.transAxes,
                color=TEXT, fontsize=9, va="center")
        ax.add_patch(FancyBboxPatch((x + 0.07, yy), width * 0.72, row_h * 0.42,
                                    boxstyle="round,pad=0.002,rounding_size=0.006",
                                    facecolor="#22354a", edgecolor="none",
                                    transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((x + 0.07, yy), width * 0.72 * value, row_h * 0.42,
                                    boxstyle="round,pad=0.002,rounding_size=0.006",
                                    facecolor=color, edgecolor="none",
                                    transform=ax.transAxes))
        ax.text(x + width * 0.83, yy + row_h * 0.28, f"{value:.0%}",
                transform=ax.transAxes, color=MUTED, fontsize=8.5, va="center")


def draw_hypnogram(ax, x, y, width, height):
    stages = np.array([0, 0, 1, 2, 2, 2, 3, 3, 2, 2, 4, 4, 2, 1, 2, 3, 3, 2, 4, 4])
    colors = [GREEN, AMBER, "#facc15", BLUE, "#c084fc"]
    xx = np.linspace(x, x + width, len(stages) + 1)
    for i, stage in enumerate(stages):
        ax.add_patch(plt.Rectangle((xx[i], y + height * stage / 5), xx[i + 1] - xx[i], height / 5,
                                   transform=ax.transAxes, facecolor=colors[stage], edgecolor=BG,
                                   linewidth=0.7))
    ax.text(x, y - 0.025, "time ->", transform=ax.transAxes, color=MUTED, fontsize=8)


def main():
    fig = plt.figure(figsize=(18, 10.5), facecolor=BG, dpi=150)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()

    ax.text(0.055, 0.935, "REALTIME SLEEP SIGNAL MONITOR", transform=ax.transAxes,
            color=TEAL, fontsize=12, fontweight="bold", va="top")
    ax.text(0.055, 0.885, "From raw EEG to an interpretable sleep-stage prediction",
            transform=ax.transAxes, color=TEXT, fontsize=25, fontweight="bold", va="top")
    ax.text(0.055, 0.842, "A 30-second epoch is processed, classified, and added to the running hypnogram.",
            transform=ax.transAxes, color=MUTED, fontsize=11, va="top")

    rounded_box(ax, 0.055, 0.53, 0.22, 0.235, "1  LIVE SIGNALS", "Six EEG channels\n3,000 samples / epoch\n100 Hz sampling rate", TEAL)
    draw_signal(ax, 0.08, 0.575, 0.17, 0.045, 1, TEAL, "EEG 1")
    draw_signal(ax, 0.08, 0.625, 0.17, 0.045, 2, CORAL, "EEG 2")
    draw_signal(ax, 0.08, 0.675, 0.17, 0.045, 3, AMBER, "EEG 3")

    rounded_box(ax, 0.32, 0.53, 0.22, 0.235, "2  PROCESSING", "Select EEG channels\nApply training mean/std\nTranspose to [channels, time]", AMBER)
    draw_signal(ax, 0.345, 0.58, 0.17, 0.11, 4, AMBER, "normalized model input")

    rounded_box(ax, 0.585, 0.53, 0.18, 0.235, "3  MODEL", "1D CNN feature extractor\nOptional BiLSTM context\nInference on CPU or CUDA", BLUE)
    for i in range(4):
        ax.add_patch(FancyBboxPatch((0.62 + i * 0.027, 0.59), 0.018, 0.10 + (i % 2) * 0.035,
                                    boxstyle="round,pad=0.002,rounding_size=0.005",
                                    transform=ax.transAxes, facecolor=BLUE, edgecolor="none", alpha=0.8))
    ax.text(0.62, 0.565, "features", transform=ax.transAxes, color=MUTED, fontsize=8.5)

    rounded_box(ax, 0.81, 0.53, 0.14, 0.235, "4  OUTPUT", "Current stage\nconfidence\nprocessing latency", CORAL)
    ax.text(0.835, 0.61, "N2", transform=ax.transAxes, color=CORAL, fontsize=27, fontweight="bold")
    ax.text(0.838, 0.58, "63% confidence", transform=ax.transAxes, color=TEXT, fontsize=9)
    ax.text(0.838, 0.55, "~12 ms", transform=ax.transAxes, color=MUTED, fontsize=9)

    arrow(ax, (0.278, 0.647), (0.315, 0.647))
    arrow(ax, (0.543, 0.647), (0.58, 0.647))
    arrow(ax, (0.768, 0.647), (0.805, 0.647))

    ax.text(0.055, 0.46, "WHAT THE MODEL SEES", transform=ax.transAxes,
            color=TEAL, fontsize=10, fontweight="bold")
    ax.text(0.055, 0.425, "Signals are preserved for visual inspection while the exact transformed input is shown beside them.",
            transform=ax.transAxes, color=TEXT, fontsize=12)

    rounded_box(ax, 0.055, 0.18, 0.42, 0.19, "PREDICTION PROBABILITIES", "The classifier returns a probability for each sleep stage.\nThe highest probability becomes the displayed current stage.", TEXT)
    probability_bars(ax, 0.08, 0.205, 0.35, 0.115)

    rounded_box(ax, 0.52, 0.18, 0.43, 0.19, "RUNNING HYPNOGRAM", "Each completed epoch is appended to the stage history,\nso transitions across the night remain easy to inspect.", GREEN)
    draw_hypnogram(ax, 0.55, 0.205, 0.36, 0.105)

    paragraph = ("This monitor replays preprocessed sleep data one 30-second epoch at a time. "
                 "For every epoch, the dashboard displays the raw EEG, the channel-selected and z-score-normalized "
                 "model input, the CNN or CNN-BiLSTM processing path, the five sleep-stage probabilities, and the "
                 "current hypnogram history. The visualization is designed to make each prediction traceable from "
                 "signal to result; it is a replay view of NPZ data rather than a direct hardware acquisition screen.")
    ax.add_patch(FancyBboxPatch((0.055, 0.055), 0.895, 0.085,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                linewidth=0.8, edgecolor="#29445c", facecolor=PANEL_2,
                                transform=ax.transAxes))
    ax.text(0.075, 0.098, paragraph, transform=ax.transAxes, color=MUTED,
            fontsize=10.2, va="center", wrap=True, linespacing=1.45)

    fig.savefig(OUTPUT, dpi=180, facecolor=BG)
    print(f"Saved {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
