import os
import glob
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


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
    x = d["x"].astype(np.float32)   # [epochs, time, channels]
    y = d["y"].astype(np.int64)     # [epochs]
    return x, y


def maybe_select_eeg(x: np.ndarray, eeg_only: bool):
    # If EEG+EOG was preprocessed, keep first 6 as EEG channels.
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


def build_class_weights(train_files):
    counts = np.zeros(5, dtype=np.float64)
    for f in train_files:
        _, y = load_subject_npz(f)
        for c in range(5):
            counts[c] += np.sum(y == c)

    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    model.train() if train_mode else model.eval()

    losses = []
    y_true_all, y_pred_all = [], []

    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
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


def main(args):
    set_seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.data_dir, "SN_*.npz")), key=subject_id_from_path)
    if not files:
        raise RuntimeError(f"No SN_*.npz found in {args.data_dir}")

    # Subject-wise split: 80 test/train split, then val from train.
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=args.seed)
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=args.seed)
    # Overall: 70 train, 10 val, 20 test subjects

    mean, std = compute_norm_stats(train_files, eeg_only=args.eeg_only)

    x0, _ = load_subject_npz(train_files[0])
    x0 = maybe_select_eeg(x0, args.eeg_only)
    in_channels = x0.shape[-1]

    train_ds = SubjectEpochDataset(train_files, mean, std, eeg_only=args.eeg_only)
    val_ds = SubjectEpochDataset(val_files, mean, std, eeg_only=args.eeg_only)
    test_ds = SubjectEpochDataset(test_files, mean, std, eeg_only=args.eeg_only)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SleepCNN(in_channels=in_channels, n_classes=5).to(device)

    class_weights = build_class_weights(train_files).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1m={tr['f1_macro']:.4f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f} f1m={va['f1_macro']:.4f}"
        )

        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.early_stop:
            print("Early stopping.")
            break

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
    out_path = os.path.join(args.out_dir, "cnn_sleep_stage_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "best_val_f1_macro": best_val_f1,
        "train_subjects": [os.path.basename(x) for x in train_files],
        "val_subjects": [os.path.basename(x) for x in val_files],
        "test_subjects": [os.path.basename(x) for x in test_files],
        "eeg_only": args.eeg_only
    }, out_path)
    print(f"\nSaved checkpoint: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to numpy_subjects with SN_*.npz")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eeg_only", action="store_true")
    args = parser.parse_args()

    main(args)