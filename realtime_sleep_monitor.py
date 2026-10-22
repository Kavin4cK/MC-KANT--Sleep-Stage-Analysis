"""Replay preprocessed sleep epochs as a live signal and sleep-stage dashboard.

Example:
    python realtime_sleep_monitor.py \
        --data_dir "C:/path/to/numpy_subjects" \
        --subject "SN_1.npz" \
        --checkpoint "checkpoints/cnn_sleep_stage_best.pt"

The NPZ files produced by iSLEEPS are replayed one 30-second epoch at a time.
Use --speed 10 to inspect a night ten times faster than real time.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

try:
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Install the dashboard dependency first: python -m pip install matplotlib") from exc

from train_cnn_sleep import SequenceSleepModel, SleepCNN, maybe_select_eeg


CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]
CLASS_COLORS = ["#52b788", "#f4a261", "#e9c46a", "#457b9d", "#c77dff"]
CHANNEL_NAMES = ["EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6"]
SAMPLE_RATE_HZ = 100
EPOCH_SECONDS = 30


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path) as data:
        if "x" not in data:
            raise KeyError(f"{path} does not contain an 'x' array")
        signals = data["x"].astype(np.float32)
        labels = data["y"].astype(np.int64) if "y" in data else None

    if signals.ndim != 3:
        raise ValueError(f"Expected x with shape [epochs, time, channels], got {signals.shape}")
    return signals, labels


def select_subject(data_dir: Path, subject: str | None) -> Path:
    if subject:
        candidate = Path(subject)
        path = candidate if candidate.is_absolute() else data_dir / candidate
        if not path.is_file():
            raise FileNotFoundError(f"Subject file not found: {path}")
        return path

    files = sorted(data_dir.glob("SN_*.npz"))
    if not files:
        raise FileNotFoundError(f"No SN_*.npz files found in {data_dir}")
    return files[0]


def build_model(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, int, str]:
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    model_arch = checkpoint.get("model_arch", "cnn")
    seq_len = int(checkpoint.get("seq_len", 1))
    n_classes = len(checkpoint.get("class_names", CLASS_NAMES))

    if model_arch == "cnn_bilstm_att" or seq_len > 1:
        model = SequenceSleepModel(
            in_channels=len(mean),
            n_classes=n_classes,
            lstm_hidden=int(checkpoint.get("lstm_hidden", 128)),
            lstm_layers=int(checkpoint.get("lstm_layers", 1)),
            attn_dim=int(checkpoint.get("attn_dim", 128)),
        )
    else:
        model = SleepCNN(in_channels=len(mean), n_classes=n_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, seq_len, model_arch


def normalize_epoch(epoch: np.ndarray, mean: np.ndarray, std: np.ndarray, eeg_only: bool) -> np.ndarray:
    selected = maybe_select_eeg(epoch[None, ...], eeg_only)[0]
    if selected.shape[-1] != len(mean):
        raise ValueError(
            f"Channel mismatch: data has {selected.shape[-1]} channels after selection, "
            f"but checkpoint expects {len(mean)}"
        )
    normalized = (selected - mean[None, :]) / np.maximum(std[None, :], 1e-8)
    return normalized.T.astype(np.float32)


def infer(
    model: torch.nn.Module,
    normalized_epochs: deque[np.ndarray],
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    if seq_len > 1:
        window = list(normalized_epochs)[-seq_len:]
        if len(window) < seq_len:
            window = [window[0]] * (seq_len - len(window)) + window
        batch = torch.from_numpy(np.stack(window)[None, ...])
    else:
        batch = torch.from_numpy(normalized_epochs[-1][None, ...])

    with torch.inference_mode():
        return torch.softmax(model(batch.to(device)), dim=1)[0].cpu().numpy()


def draw_dashboard(
    axes: tuple,
    raw_lines: list,
    normalized_lines: list,
    probabilities: np.ndarray,
    history: list[int],
    raw_epoch: np.ndarray,
    normalized_epoch: np.ndarray,
    epoch_index: int,
    subject_path: Path,
    model_arch: str,
    seq_len: int,
    device: torch.device,
    true_label: int | None,
    elapsed_ms: float,
) -> None:
    raw_ax, norm_ax, prob_ax, info_ax, history_ax = axes
    time_axis = np.arange(raw_epoch.shape[0]) / SAMPLE_RATE_HZ
    raw_scale = max(float(np.percentile(np.abs(raw_epoch), 95)), 1e-6)
    raw_offset = raw_scale * 2.4

    for channel, line in enumerate(raw_lines):
        line.set_data(time_axis, raw_epoch[:, channel] + channel * raw_offset)
    raw_ax.set_xlim(0, max(time_axis[-1], EPOCH_SECONDS))
    raw_ax.set_ylim(-raw_scale, max(raw_epoch.shape[1] - 1, 1) * raw_offset + raw_scale)
    raw_ax.set_yticks([channel * raw_offset for channel in range(raw_epoch.shape[1])])
    raw_ax.set_yticklabels(CHANNEL_NAMES[: raw_epoch.shape[1]])
    raw_ax.set_title(f"Live EEG window | epoch {epoch_index + 1}")

    for channel, line in enumerate(normalized_lines):
        line.set_data(time_axis, normalized_epoch[channel] + channel * 4.0)
    norm_ax.set_xlim(0, max(time_axis[-1], EPOCH_SECONDS))
    norm_ax.set_ylim(-3, max(normalized_epoch.shape[0] - 1, 1) * 4.0 + 3)
    norm_ax.set_yticks([channel * 4.0 for channel in range(normalized_epoch.shape[0])])
    norm_ax.set_yticklabels(CHANNEL_NAMES[: normalized_epoch.shape[0]])
    norm_ax.set_title("Model input after channel selection + z-score")

    prob_ax.clear()
    order = np.argsort(probabilities)
    prob_ax.barh(
        np.arange(len(probabilities)), probabilities[order],
        color=[CLASS_COLORS[i] for i in order],
    )
    prob_ax.set_yticks(np.arange(len(probabilities)))
    prob_ax.set_yticklabels([CLASS_NAMES[i] for i in order])
    prob_ax.set_xlim(0, 1)
    prob_ax.set_xlabel("Probability")
    prob_ax.set_title(f"Prediction: {CLASS_NAMES[int(np.argmax(probabilities))]}")
    for row, class_index in enumerate(order):
        prob_ax.text(probabilities[class_index] + 0.02, row, f"{probabilities[class_index]:.1%}", va="center")

    info_ax.clear()
    info_ax.axis("off")
    true_text = "unknown" if true_label is None else CLASS_NAMES[true_label]
    info_ax.text(
        0.02, 0.96,
        "PROCESSING\n\n"
        f"source       {subject_path.name}\n"
        f"epoch        {epoch_index + 1}\n"
        f"sample rate  {SAMPLE_RATE_HZ} Hz\n"
        f"window       {EPOCH_SECONDS} s ({raw_epoch.shape[0]:,} samples)\n"
        f"channels     {raw_epoch.shape[1]} -> {normalized_epoch.shape[0]}\n"
        f"normalizer   checkpoint train mean/std\n"
        f"model        {model_arch} / context {seq_len} epoch(s)\n"
        f"device       {device}\n"
        f"inference    {elapsed_ms:.1f} ms\n"
        f"label        {true_text}",
        va="top", family="monospace", fontsize=9,
    )

    history_ax.clear()
    history_ax.imshow(
        np.array(history, dtype=np.int16)[None, :],
        aspect="auto", interpolation="nearest", vmin=0, vmax=len(CLASS_NAMES) - 1,
        cmap=ListedColormap(CLASS_COLORS),
    )
    history_ax.set_yticks([])
    history_ax.set_xlabel("Completed epochs")
    history_ax.set_title("Hypnogram preview")
    history_ax.set_xticks(np.linspace(0, max(len(history) - 1, 0), min(6, len(history))).astype(int) if history else [])

    plt.gcf().canvas.draw_idle()
    plt.pause(0.001)


def run(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    subject_path = select_subject(Path(args.data_dir), args.subject)
    raw_signals, labels = load_subject(subject_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    eeg_only = bool(checkpoint.get("eeg_only", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    model, seq_len, model_arch = build_model(checkpoint, device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    if list(class_names) != CLASS_NAMES:
        raise ValueError(f"Unsupported class_names in checkpoint: {class_names}")

    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 0.65], width_ratios=[1.2, 1.2, 0.8])
    raw_ax = figure.add_subplot(grid[0, :2])
    norm_ax = figure.add_subplot(grid[1, :2])
    prob_ax = figure.add_subplot(grid[0, 2])
    info_ax = figure.add_subplot(grid[1, 2])
    history_ax = figure.add_subplot(grid[2, :])
    figure.suptitle("Sleep staging monitor", fontsize=16, fontweight="bold")

    channel_count = raw_signals.shape[-1] if not eeg_only else min(raw_signals.shape[-1], 6)
    raw_lines = [raw_ax.plot([], [], lw=0.7)[0] for _ in range(channel_count)]
    normalized_lines = [norm_ax.plot([], [], lw=0.7)[0] for _ in range(len(mean))]
    normalized_epochs: deque[np.ndarray] = deque(maxlen=max(seq_len, 1))
    history: list[int] = []

    plt.ion()
    for epoch_index, raw_epoch in enumerate(raw_signals):
        if not plt.fignum_exists(figure.number):
            break
        started = perf_counter()
        normalized_epoch = normalize_epoch(raw_epoch, mean, std, eeg_only)
        normalized_epochs.append(normalized_epoch)
        probabilities = infer(model, normalized_epochs, seq_len, device)
        elapsed_ms = (perf_counter() - started) * 1000
        predicted = int(np.argmax(probabilities))
        history.append(predicted)
        true_label = int(labels[epoch_index]) if labels is not None and epoch_index < len(labels) else None
        draw_dashboard(
            (raw_ax, norm_ax, prob_ax, info_ax, history_ax),
            raw_lines, normalized_lines, probabilities, history,
            raw_epoch[:, :channel_count], normalized_epoch, epoch_index,
            subject_path, model_arch, seq_len, device, true_label, elapsed_ms,
        )
        wait_seconds = EPOCH_SECONDS / max(args.speed, 1e-6)
        deadline = perf_counter() + wait_seconds
        while perf_counter() < deadline and plt.fignum_exists(figure.number):
            plt.pause(min(0.1, max(deadline - perf_counter(), 0)))

    plt.ioff()
    if plt.fignum_exists(figure.number):
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay NPZ sleep signals with live model predictions")
    parser.add_argument("--data_dir", required=True, help="Folder containing SN_*.npz")
    parser.add_argument("--subject", help="Subject filename, for example SN_1.npz; defaults to the first subject")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_sleep_stage_best.pt")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed; 1 is real-time, 10 is ten times faster")
    parser.add_argument("--force_cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
