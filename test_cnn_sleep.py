import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from train_cnn_sleep import (
    SubjectEpochDataset,
    SequenceSubjectDataset,
    SleepCNN,
    SequenceSleepModel,
    resolve_subject_files,
    subject_id_from_path,
)


CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]


def run_eval(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)

            losses.append(loss.item())
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
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    files = sorted(resolve_subject_files(args.data_dir), key=subject_id_from_path)
    train_files, test_files = train_test_split(
        files,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    if "mean" not in ckpt or "std" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain: model_state_dict, mean, std")

    mean = ckpt["mean"].astype(np.float32)
    std = ckpt["std"].astype(np.float32)

    eeg_only = ckpt.get("eeg_only", args.eeg_only)
    n_classes = args.n_classes
    model_arch = ckpt.get("model_arch", "cnn")
    seq_len = ckpt.get("seq_len", args.seq_len)

    if model_arch == "cnn_bilstm_att" or seq_len > 1:
        test_ds = SequenceSubjectDataset(
            test_files,
            mean,
            std,
            seq_len=seq_len,
            eeg_only=eeg_only,
        )
    else:
        test_ds = SubjectEpochDataset(test_files, mean, std, eeg_only=eeg_only)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    in_channels = len(mean)
    if model_arch == "cnn_bilstm_att" or seq_len > 1:
        lstm_hidden = ckpt.get("lstm_hidden", args.lstm_hidden)
        lstm_layers = ckpt.get("lstm_layers", args.lstm_layers)
        attn_dim = ckpt.get("attn_dim", args.attn_dim)
        model = SequenceSleepModel(
            in_channels=in_channels,
            n_classes=n_classes,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            attn_dim=attn_dim,
        )
    else:
        model = SleepCNN(in_channels=in_channels, n_classes=n_classes)
    model.load_state_dict(ckpt["model_state_dict"])

    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    out = run_eval(model, test_loader, criterion, device)

    print(f"Testing on device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Subjects -> train: {len(train_files)}, test: {len(test_files)}")
    print(f"Epoch samples -> test: {len(test_ds)}")

    print("\nTest Metrics")
    print(f"Loss: {out['loss']:.4f}")
    print(f"Accuracy: {out['acc']:.4f}")
    print(f"Macro F1: {out['f1_macro']:.4f}")

    print("\nClassification Report (0=W, 1=N1, 2=N2, 3=N3, 4=R)")
    print(classification_report(out["y_true"], out["y_pred"], digits=4))

    print("Confusion Matrix")
    print(confusion_matrix(out["y_true"], out["y_pred"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved SleepCNN checkpoint")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder with SN_*.npz files.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/cnn_sleep_stage_best.pt",
        help="Path to saved checkpoint .pt file",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--eeg_only", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    main(args)
