import os
import glob
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_id_from_path(path: str) -> int:
    name = os.path.basename(path).replace(".npz", "")
    return int(name.split("_")[1])


def load_subject_npz(path: str):
    d = np.load(path)
    if "x" not in d or "y" not in d:
        raise KeyError(f"Expected keys 'x' and 'y' in {path}; found {list(d.keys())}")

    x = d["x"].astype(np.float32)  # [epochs, time, channels]
    y = d["y"].astype(np.int64)    # [epochs]

    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3 in {path}, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y.ndim == 1 in {path}, got shape {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched epochs in {path}: x={x.shape[0]}, y={y.shape[0]}")

    return x, y


def maybe_select_eeg(x: np.ndarray, eeg_only: bool):
    # If EEG+EOG channels exist, keep first 6 as EEG channels.
    if eeg_only and x.shape[-1] > 6:
        x = x[:, :, :6]
    return x


def compute_norm_stats(train_files, eeg_only: bool):
    sum_c, sumsq_c, total = None, None, 0

    for f in train_files:
        x, _ = load_subject_npz(f)
        x = maybe_select_eeg(x, eeg_only)
        flat = x.reshape(-1, x.shape[-1])  # [epochs*time, channels]

        if sum_c is None:
            sum_c = flat.sum(axis=0)
            sumsq_c = (flat ** 2).sum(axis=0)
        else:
            sum_c += flat.sum(axis=0)
            sumsq_c += (flat ** 2).sum(axis=0)

        total += flat.shape[0]

    mean = sum_c / total
    var = (sumsq_c / total) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class SubjectEpochDataset(Dataset):
    def __init__(self, files, mean, std, eeg_only=True):
        self.files = files
        self.mean = mean
        self.std = std
        self.eeg_only = eeg_only
        self.mapping = []
        self.cache = {}

        for i, f in enumerate(self.files):
            _, y = load_subject_npz(f)
            for e in range(len(y)):
                self.mapping.append((i, e))

    def __len__(self):
        return len(self.mapping)

    def _load_cached(self, i):
        if i not in self.cache:
            x, y = load_subject_npz(self.files[i])
            x = maybe_select_eeg(x, self.eeg_only)
            x = (x - self.mean[None, None, :]) / self.std[None, None, :]
            x = np.transpose(x, (0, 2, 1))  # [epochs, channels, time] for Conv1d
            self.cache[i] = (x, y)
        return self.cache[i]

    def __getitem__(self, idx):
        s_idx, e_idx = self.mapping[idx]
        x, y = self._load_cached(s_idx)
        return torch.from_numpy(x[e_idx]), torch.tensor(y[e_idx], dtype=torch.long)


class SequenceSubjectDataset(Dataset):
    def __init__(self, files, mean, std, seq_len: int, eeg_only=True):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        self.files = files
        self.mean = mean
        self.std = std
        self.seq_len = seq_len
        self.eeg_only = eeg_only
        self.mapping = []
        self.cache = {}

        for i, f in enumerate(self.files):
            _, y = load_subject_npz(f)
            max_start = len(y) - self.seq_len
            for s in range(max_start + 1):
                self.mapping.append((i, s))

    def __len__(self):
        return len(self.mapping)

    def _load_cached(self, i):
        if i not in self.cache:
            x, y = load_subject_npz(self.files[i])
            x = maybe_select_eeg(x, self.eeg_only)
            x = (x - self.mean[None, None, :]) / self.std[None, None, :]
            x = np.transpose(x, (0, 2, 1))  # [epochs, channels, time]
            self.cache[i] = (x, y)
        return self.cache[i]

    def __getitem__(self, idx):
        s_idx, start = self.mapping[idx]
        x, y = self._load_cached(s_idx)

        seq_x = x[start:start + self.seq_len]
        # Use center epoch label to represent the 5-epoch window.
        center = start + (self.seq_len // 2)
        target = y[center]
        return torch.from_numpy(seq_x), torch.tensor(target, dtype=torch.long)


class SleepCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        # x: [batch, time, hidden]
        scores = self.v(torch.tanh(self.proj(x))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class SequenceSleepModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels)

        # BiLSTM adds temporal context across consecutive epochs.
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Attention focuses on the most informative epochs in the window.
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2, attn_dim=attn_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

        self.debug_shapes = False

    def forward(self, x):
        # x: [batch, seq_len, channels, time]
        input_shape = tuple(x.shape)
        bsz, seq_len, channels, time = x.shape

        x = x.view(bsz * seq_len, channels, time)
        feats = self.cnn(x)
        feats = feats.view(bsz, seq_len, -1)

        lstm_out, _ = self.lstm(feats)
        context, attn_w = self.attn(lstm_out)
        logits = self.classifier(context)

        if self.debug_shapes:
            print(f"input: {input_shape}")
            print(f"cnn_embeddings: {tuple(feats.shape)}")
            print(f"lstm_out: {tuple(lstm_out.shape)}")
            print(f"attn_weights: {tuple(attn_w.shape)}")
            print(f"context: {tuple(context.shape)}")
            print(f"logits: {tuple(logits.shape)}")

        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.gather(0, targets.squeeze(1))
        elif self.alpha is not None:
            alpha_t = self.alpha
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_class_weights(train_files, n_classes=5):
    counts = np.zeros(n_classes, dtype=np.float64)
    for f in train_files:
        _, y = load_subject_npz(f)
        for c in range(n_classes):
            counts[c] += np.sum(y == c)

    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def parse_focal_alpha(value: str, n_classes: int):
    if value is None or value == "":
        return None

    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != n_classes:
        raise ValueError("focal_alpha must be a single value or a list with n_classes values")
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    model.train() if train_mode else model.eval()

    losses = []
    y_true_all, y_pred_all = [], []

    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            if getattr(model, "debug_shapes", False):
                model.debug_shapes = False
            loss = criterion(logits, yb)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(loss.item())
            preds = logits.argmax(dim=1)
            y_true_all.append(yb.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def resolve_subject_files(data_dir: str):
    pattern = os.path.join(data_dir, "SN_*.npz")
    files = sorted(glob.glob(pattern), key=subject_id_from_path)
    if not files:
        raise RuntimeError(
            f"No subject files found with pattern {pattern}. "
            f"Use the folder that contains SN_1.npz ... SN_100.npz"
        )
    return files


def main(args):
    set_seed(args.seed)

    files = resolve_subject_files(args.data_dir)
    print(f"Found {len(files)} subject files in: {args.data_dir}")

    # Subject-wise split to avoid leakage across train/test.
    train_files, test_files = train_test_split(
        files,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    if args.val_size > 0:
        val_ratio_from_train = args.val_size / (1.0 - args.test_size)
        train_files, val_files = train_test_split(
            train_files,
            test_size=val_ratio_from_train,
            random_state=args.seed,
            shuffle=True,
        )
    else:
        val_files = []

    print(
        f"Split subjects -> train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}"
    )

    mean, std = compute_norm_stats(train_files, eeg_only=args.eeg_only)

    x0, _ = load_subject_npz(train_files[0])
    x0 = maybe_select_eeg(x0, args.eeg_only)
    in_channels = x0.shape[-1]

    if args.seq_len > 1:
        train_ds = SequenceSubjectDataset(
            train_files,
            mean,
            std,
            seq_len=args.seq_len,
            eeg_only=args.eeg_only,
        )
        test_ds = SequenceSubjectDataset(
            test_files,
            mean,
            std,
            seq_len=args.seq_len,
            eeg_only=args.eeg_only,
        )
        val_ds = (
            SequenceSubjectDataset(
                val_files,
                mean,
                std,
                seq_len=args.seq_len,
                eeg_only=args.eeg_only,
            )
            if val_files
            else None
        )
    else:
        train_ds = SubjectEpochDataset(train_files, mean, std, eeg_only=args.eeg_only)
        test_ds = SubjectEpochDataset(test_files, mean, std, eeg_only=args.eeg_only)
        val_ds = SubjectEpochDataset(val_files, mean, std, eeg_only=args.eeg_only) if val_files else None

    val_epoch_count = len(val_ds) if val_ds is not None else 0
    print(f"Epoch samples -> train: {len(train_ds)}, val: {val_epoch_count}, test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seq_len > 1:
        model = SequenceSleepModel(
            in_channels=in_channels,
            n_classes=args.n_classes,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            attn_dim=args.attn_dim,
        ).to(device)
    else:
        model = SleepCNN(in_channels=in_channels, n_classes=args.n_classes).to(device)

    if hasattr(model, "debug_shapes") and args.debug_shapes:
        model.debug_shapes = True

    class_weights = build_class_weights(train_files, n_classes=args.n_classes).to(device)

    # Focal loss addresses class imbalance by focusing on harder examples.
    alpha = parse_focal_alpha(args.focal_alpha, args.n_classes)
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(device)
    elif alpha is None:
        alpha = class_weights

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_count = 0

    print(f"Training on device: {device}")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        if val_loader is not None:
            va = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1m={tr['f1_macro']:.4f} | "
                f"val loss={va['loss']:.4f} acc={va['acc']:.4f} f1m={va['f1_macro']:.4f}"
            )

            score = va["f1_macro"]
            if score > best_val_f1:
                best_val_f1 = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= args.early_stop:
                print("Early stopping triggered.")
                break
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1m={tr['f1_macro']:.4f}"
            )
            best_val_f1 = tr["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te = run_epoch(model, test_loader, criterion, optimizer, device, train_mode=False)

    print("\nTest Metrics")
    print(f"Loss: {te['loss']:.4f}")
    print(f"Accuracy: {te['acc']:.4f}")
    print(f"Macro F1: {te['f1_macro']:.4f}")

    print("\nClassification Report (0=W, 1=N1, 2=N2, 3=N3, 4=R)")
    print(classification_report(te["y_true"], te["y_pred"], digits=4))
    print("Confusion Matrix")
    print(confusion_matrix(te["y_true"], te["y_pred"]))

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "cnn_sleep_stage_best.pt")
    metrics_path = os.path.join(args.out_dir, "cnn_sleep_metrics.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "best_val_f1_macro": best_val_f1,
            "test_loss": te["loss"],
            "test_accuracy": te["acc"],
            "test_f1_macro": te["f1_macro"],
            "class_names": CLASS_NAMES,
            "train_subjects": [os.path.basename(x) for x in train_files],
            "val_subjects": [os.path.basename(x) for x in val_files],
            "test_subjects": [os.path.basename(x) for x in test_files],
            "eeg_only": args.eeg_only,
            "model_arch": "cnn_bilstm_att" if args.seq_len > 1 else "cnn",
            "seq_len": args.seq_len,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "attn_dim": args.attn_dim,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
        },
        ckpt_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_f1_macro": best_val_f1,
                "test_loss": te["loss"],
                "test_accuracy": te["acc"],
                "test_f1_macro": te["f1_macro"],
            },
            f,
            indent=2,
        )

    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for EEG sleep stage classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder with SN_*.npz files (subject-level).",
    )
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=str, default="")
    parser.add_argument("--debug_shapes", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.0)
    parser.add_argument("--eeg_only", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    if not (0.0 <= args.val_size < 1.0):
        raise ValueError("val_size must be between 0 (inclusive) and 1")
    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1")

    main(args)
