"""
Phase 6 — Training Dynamics & Optimization Loop
================================================
Implements:
  • AdamW optimizer with cosine annealing + warm restarts (CosineAnnealingWarmRestarts)
  • Focal Loss for class-imbalance mitigation  (imported from model.py)
  • Supervised Contrastive pre-training phase
  • Patient-wise 10-fold stratified cross-validation (StratifiedKFold)
    ensuring zero data leakage between splits
  • Full end-to-end fine-tuning phase on N-epoch sequences
  • Per-fold evaluation: accuracy, macro-F1, per-class F1, confusion matrix
  • Checkpoint saving for best val macro-F1 per fold

Usage
-----
From command line:
    python -m mc_kant.train \
        --data_dir  /path/to/numpy_subjects \
        --out_dir   ./mc_kant_checkpoints \
        --epochs_pretrain 30 \
        --epochs_finetune 50 \
        --seq_len 20 \
        --batch_size 8 \
        --device cuda

Programmatic:
    from mc_kant.train import train_kfold
    train_kfold(args)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from .dataset import (
    compute_norm_stats,
    get_subject_labels,
    resolve_npz_files,
    NPZ_X_KEY,
    NPZ_Y_KEY,
)
from .model import MCKANT, FocalLoss
from .contrastive import SupervisedContrastiveLoss, generate_two_views


# ---------------------------------------------------------------------------
# Multi-channel dataset: uses ALL available NPZ channels as one EEG stream
# ---------------------------------------------------------------------------

class MultiChannelDataset(Dataset):
    """
    Dataset that returns ALL channels from SN_*.npz files as a single
    multi-channel EEG stream: shape [C, T] per epoch (or [N, C, T] for seq).

    Uses mmap_mode='r' during __init__ for fast epoch counting (reads only
    the NPZ header, not the full 10 MB of signal data).  Full data is loaded
    lazily and cached on first __getitem__ access.

    Args:
        files        : List of SN_*.npz paths
        mean         : [C] per-channel mean (length determines min_channels)
        std          : [C] per-channel std
        seq_len      : 1 = single epoch; >1 = sliding window sequence
        epoch_counts : Optional pre-computed {filepath: n_epochs} dict to
                       skip file scanning entirely (reuse across datasets in
                       the same fold)
    """

    def __init__(
        self,
        files: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        seq_len: int = 1,
        epoch_counts: Optional[Dict[str, int]] = None,
    ):
        self.files   = files
        self.mean    = mean
        self.std     = std
        self.seq_len = seq_len
        self.mapping: List[Tuple[int, int]] = []
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for fi, f in enumerate(files):
            if epoch_counts is not None and f in epoch_counts:
                # Reuse pre-computed count — zero file I/O
                n = epoch_counts[f]
            else:
                # mmap_mode='r': reads only the NPZ header/metadata,
                # NOT the full signal array. ~100x faster than np.load(f)
                d = np.load(f, mmap_mode='r', allow_pickle=False)
                n = d[NPZ_Y_KEY].shape[0]
                if epoch_counts is not None:
                    epoch_counts[f] = n   # cache for reuse

            if seq_len <= 1:
                self.mapping.extend((fi, e) for e in range(n))
            else:
                self.mapping.extend((fi, s) for s in range(n - seq_len + 1))

    def __len__(self) -> int:
        return len(self.mapping)

    def _load_cached(self, fi: int) -> Tuple[np.ndarray, np.ndarray]:
        """Full data load (only on first access for each subject file)."""
        if fi not in self.cache:
            # Full load — necessary for signal data
            d = np.load(self.files[fi], allow_pickle=False)
            x = d[NPZ_X_KEY].astype(np.float32)   # [E, T, C]
            min_c = len(self.mean)                 # clip to consistent C
            x = x[:, :, :min_c]                    # [E, T, C_min]
            x = (x - self.mean[None, None, :]) / (self.std[None, None, :] + 1e-8)
            x = np.transpose(x, (0, 2, 1))         # [E, C_min, T]
            y = d[NPZ_Y_KEY].astype(np.int64)
            self.cache[fi] = (x, y)
        return self.cache[fi]

    def __getitem__(self, idx: int):
        fi, start = self.mapping[idx]
        x, y = self._load_cached(fi)

        if self.seq_len <= 1:
            return (
                torch.from_numpy(x[start].copy()),              # [C, T]
                torch.tensor(y[start], dtype=torch.long),
            )
        else:
            center = start + self.seq_len // 2
            return (
                torch.from_numpy(x[start : start + self.seq_len].copy()),  # [N, C, T]
                torch.tensor(y[center], dtype=torch.long),
            )



# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_amp_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name in ("off", "none", "false"):
        return torch.float32
    raise ValueError(f"Unsupported amp dtype: {name}. Use bf16, fp16, or off.")


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(device: torch.device, amp_dtype: torch.dtype) -> torch.amp.GradScaler:
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler(device="cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _print_cuda_info() -> None:
    if not torch.cuda.is_available():
        print("[MC-KANT] CUDA is not available.")
        return

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    total_gb = props.total_memory / (1024 ** 3)
    print(
        f"[MC-KANT] CUDA GPU: {props.name} "
        f"({total_gb:.1f} GB, capability {props.major}.{props.minor})"
    )
    print(f"[MC-KANT] torch={torch.__version__} cuda={torch.version.cuda}")


def _model_for_state(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _resume_path(args: argparse.Namespace) -> str:
    return getattr(args, "resume_path", None) or os.path.join(args.out_dir, "mc_kant_resume.pt")


def _load_resume_state(args: argparse.Namespace) -> Optional[Dict]:
    if not getattr(args, "resume", True):
        return None

    path = _resume_path(args)
    if not os.path.exists(path):
        return None

    print(f"[MC-KANT] Resuming from: {path}", flush=True)
    return torch.load(path, map_location="cpu")


def _save_resume_state(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _delete_resume_state(args: argparse.Namespace) -> None:
    path = _resume_path(args)
    if os.path.exists(path):
        os.remove(path)


def _rng_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Class weight computation for Focal Loss α
# ---------------------------------------------------------------------------

def build_class_weights(files: List[str], n_classes: int = 5) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training subject files.
    Uses mmap_mode='r' for fast label-only access.
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for f in files:
        d = np.load(f, mmap_mode='r', allow_pickle=False)
        y = d[NPZ_Y_KEY].astype(np.int64)
        for c in range(n_classes):
            counts[c] += np.sum(y == c)

    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Single-epoch pre-training step (SupCon)
# ---------------------------------------------------------------------------

def _pretrain_step(
    model: MCKANT,
    batch: Tuple[torch.Tensor, ...],
    supcon_loss: SupervisedContrastiveLoss,
    focal_loss: FocalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    contrastive_weight: float = 0.5,
) -> Tuple[float, float, float]:
    """
    One gradient step during contrastive pre-training with AMP.
    Returns: (total_loss, focal_loss_val, supcon_loss_val)
    """
    x, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

    x_v1, x_v2 = generate_two_views(x)

    with _autocast_context(device, amp_enabled, amp_dtype):
        logits1, z1 = model(x_v1, contrastive_mode=True)
        logits2, z2 = model(x_v2, contrastive_mode=True)
        fl = (focal_loss(logits1, labels) + focal_loss(logits2, labels)) / 2.0
        features = torch.stack([z1, z2], dim=1)
        cl = supcon_loss(features, labels)
        loss = fl + contrastive_weight * cl

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), fl.item(), cl.item()



# ---------------------------------------------------------------------------
# Sequence fine-tuning epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: MCKANT,
    loader: DataLoader,
    focal_loss: FocalLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    train_mode: bool = True,
) -> Dict:
    """
    Run one full pass over the DataLoader in sequence fine-tuning mode with AMP.
    """
    model.train() if train_mode else model.eval()

    losses: List[float] = []
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    with torch.set_grad_enabled(train_mode):
        for batch_i, batch in enumerate(loader):
            x, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            with _autocast_context(device, amp_enabled, amp_dtype):
                logits = model(x, contrastive_mode=False)
                B, N, n_cls = logits.shape
                center_logits = logits[:, N // 2, :]
                loss = focal_loss(center_logits, labels)

            if train_mode and scaler is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            elif train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(loss.item())
            preds = center_logits.detach().argmax(dim=1)
            y_true_all.append(labels.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }



# ---------------------------------------------------------------------------
# Patient-wise stratified 10-fold cross-validation
# ---------------------------------------------------------------------------

def train_kfold(args: argparse.Namespace) -> None:
    """
    Run patient-wise 10-fold stratified cross-validation.

    Each fold:
      1. Split subjects strictly (no leakage): train / val / test sets
      2. Compute per-channel norm stats from training subjects only
      3. Pre-train MC-KANT with SupCon + Focal on single epochs
      4. Fine-tune MC-KANT end-to-end on N-epoch sequences
      5. Evaluate on test subjects → save checkpoint + metrics JSON

    Args:
        args: argparse.Namespace with all training hyperparameters.
    """
    set_seed(args.seed)

    if getattr(args, "tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if getattr(args, "require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was required but is not available. On the A100 server, install a CUDA "
            "PyTorch wheel and run from a GPU-enabled environment."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[MC-KANT] Device: {device}")
    _print_cuda_info()

    amp_dtype = _get_amp_dtype(getattr(args, "amp_dtype", "bf16"))
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    if amp_enabled:
        print(f"[MC-KANT] AMP enabled: {args.amp_dtype}")

    # ---- Resolve subject files ------------------------------------------
    all_files = resolve_npz_files(args.data_dir)
    print(f"[MC-KANT] Found {len(all_files)} subjects in: {args.data_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    resume_path = _resume_path(args)
    resume_state = _load_resume_state(args)
    if resume_state is not None:
        _restore_rng_state(resume_state.get("rng_state"))

    # Stratification labels: majority sleep stage per subject
    strat_labels = get_subject_labels(all_files)

    # ---- 10-fold StratifiedKFold on subjects ----------------------------
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_splits = list(skf.split(np.arange(len(all_files)), strat_labels))
    fold_metrics: List[Dict] = resume_state.get("fold_metrics", []) if resume_state else []
    resume_fold_idx = int(resume_state.get("fold_idx", 0)) if resume_state else 0

    for fold_idx, (train_val_idx, test_idx) in enumerate(fold_splits):
        if fold_idx < resume_fold_idx:
            print(f"[MC-KANT] Skipping completed fold {fold_idx + 1}.", flush=True)
            continue

        active_resume = resume_state if (resume_state and fold_idx == resume_fold_idx) else None
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1} / {args.n_folds}")
        print(f"{'='*60}")

        # Split train_val → train (90%) + val (10%)
        test_files = [all_files[i] for i in test_idx]
        train_val_files = [all_files[i] for i in train_val_idx]
        strat_tv = strat_labels[train_val_idx]

        # Inner 90/10 split for early stopping validation
        inner_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        train_local_idx, val_local_idx = next(inner_skf.split(
            np.arange(len(train_val_files)), strat_tv
        ))
        train_files = [train_val_files[i] for i in train_local_idx]
        val_files = [train_val_files[i] for i in val_local_idx]

        print(
            f"  Subjects → train: {len(train_files)}, "
            f"val: {len(val_files)}, test: {len(test_files)}"
        )

        # ---- Normalisation stats from training subjects only ------------
        print("  Computing normalisation stats...", flush=True)
        mean, std = compute_norm_stats(train_files)
        n_channels = mean.shape[0]   # number of channels in NPZ (e.g. 6)
        print(f"  Channels: {n_channels}, done.", flush=True)

        # Shared epoch-count cache: populated during first dataset init,
        # reused by subsequent inits for the same fold — eliminates 3x redundant I/O
        epoch_counts: Dict[str, int] = {}
        # ---- Datasets — use ALL channels as single multi-channel stream -
        # Pre-training: single epochs → x: [B, C, T]
        print("  Building datasets...", flush=True)
        pretrain_ds  = MultiChannelDataset(train_files, mean, std, seq_len=1,           epoch_counts=epoch_counts)
        seq_train_ds = MultiChannelDataset(train_files, mean, std, seq_len=args.seq_len, epoch_counts=epoch_counts)
        seq_val_ds   = MultiChannelDataset(val_files,   mean, std, seq_len=args.seq_len, epoch_counts=epoch_counts)
        seq_test_ds  = MultiChannelDataset(test_files,  mean, std, seq_len=args.seq_len, epoch_counts=epoch_counts)
        print(f"  Datasets ready: pretrain={len(pretrain_ds):,}  seq_train={len(seq_train_ds):,}  val={len(seq_val_ds):,}  test={len(seq_test_ds):,}", flush=True)

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": bool(getattr(args, "pin_memory", True) and torch.cuda.is_available()),
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))

        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=args.batch_size,
            shuffle=True, **loader_kwargs,
        )
        # Sequence Phase B: each item is [N, C, T] → flattened to [B*N, C, T] in model
        # → use smaller batch to avoid OOM: seq_batch = batch // seq_len (min 4)
        seq_batch_size = int(getattr(args, "seq_batch_size", 0) or max(16, args.batch_size // args.seq_len))
        print(f"  Batch sizes: pretrain={args.batch_size}, seq_finetune={seq_batch_size}", flush=True)

        seq_train_loader = DataLoader(
            seq_train_ds, batch_size=seq_batch_size,
            shuffle=True, **loader_kwargs,
        )
        seq_val_loader = DataLoader(
            seq_val_ds, batch_size=seq_batch_size,
            shuffle=False, **loader_kwargs,
        )
        seq_test_loader = DataLoader(
            seq_test_ds, batch_size=seq_batch_size,
            shuffle=False, **loader_kwargs,
        )

        # ---- Build model — all channels fed as single EEG stream --------
        # eog_in=0, emg_in=0: single EpochFeatureExtractor handles all C channels
        model = MCKANT(
            eeg_in_channels=n_channels,  # treat all 6 channels as one EEG stream
            eog_in_channels=0,
            emg_in_channels=0,
            fusion_dim=args.fusion_dim,
            seq_len=args.seq_len,
            unet_base_filters=args.unet_base_filters,
            unet_levels=args.unet_levels,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            proj_dim=args.proj_dim,
            kan_hidden_dims=args.kan_hidden_dims,
            kan_grid_size=args.kan_grid_size,
            n_classes=5,
            dropout=args.dropout,
        ).to(device)

        param_info = model.param_count()
        print(f"  Parameters: {param_info['total']:,} total")
        for k, v in param_info.items():
            if k != "total":
                print(f"    {k}: {v:,}")

        if getattr(args, "compile_model", False) and hasattr(torch, "compile"):
            print("  Compiling model with torch.compile...", flush=True)
            model = torch.compile(model)

        # ---- Loss functions -----------------------------------------------
        class_weights = build_class_weights(train_files).to(device)
        focal = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)
        supcon = SupervisedContrastiveLoss(temperature=args.supcon_temp)

        scaler = _make_grad_scaler(device, amp_dtype)

        if active_resume and active_resume.get("model_state_dict") is not None:
            _model_for_state(model).load_state_dict(active_resume["model_state_dict"])
            if active_resume.get("scaler_state_dict") is not None:
                try:
                    scaler.load_state_dict(active_resume["scaler_state_dict"])
                except Exception:
                    print("  Resume scaler state could not be restored; continuing with a fresh scaler.", flush=True)

        def save_resume(
            phase: str,
            next_epoch: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            best_val_f1_value: float = -1.0,
            best_state_value: Optional[Dict] = None,
            patience_value: int = 0,
        ) -> None:
            _save_resume_state(
                resume_path,
                {
                    "fold_idx": fold_idx,
                    "phase": phase,
                    "next_epoch": next_epoch,
                    "model_state_dict": _model_for_state(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_f1": best_val_f1_value,
                    "best_state": best_state_value,
                    "patience_count": patience_value,
                    "fold_metrics": fold_metrics,
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                    "rng_state": _rng_state(),
                },
            )

        # ---- Phase A: Contrastive pre-training ---------------------------
        resume_phase = active_resume.get("phase") if active_resume else None
        start_pretrain_epoch = 1
        if active_resume and resume_phase == "pretrain":
            start_pretrain_epoch = int(active_resume.get("next_epoch", 1))

        run_pretrain = args.epochs_pretrain > 0 and resume_phase not in ("finetune", "test", "fold_complete")
        if run_pretrain and start_pretrain_epoch <= args.epochs_pretrain:
            print(f"\n  [Phase A] Contrastive pre-training for {args.epochs_pretrain} epochs", flush=True)

            # AdamW with cosine annealing + warm restarts
            opt_pt = torch.optim.AdamW(
                model.parameters(), lr=args.lr_pretrain,
                weight_decay=args.weight_decay,
            )
            sched_pt = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_pt,
                T_0=max(1, args.epochs_pretrain // args.cosine_t0_divisor),
                T_mult=args.cosine_t_mult,
                eta_min=args.lr_pretrain * 1e-3,
            )
            if active_resume and resume_phase == "pretrain":
                if active_resume.get("optimizer_state_dict") is not None:
                    opt_pt.load_state_dict(active_resume["optimizer_state_dict"])
                if active_resume.get("scheduler_state_dict") is not None:
                    sched_pt.load_state_dict(active_resume["scheduler_state_dict"])
                print(f"  Resuming Phase A at epoch {start_pretrain_epoch}/{args.epochs_pretrain}", flush=True)

            for ep in range(start_pretrain_epoch, args.epochs_pretrain + 1):
                total_loss = focal_loss_sum = cl_sum = n_batches = 0
                model.train()
                t_ep = time.time()
                try:
                    for bi, batch in enumerate(pretrain_loader):
                        tl, fl_val, cl_val = _pretrain_step(
                            model, batch, supcon, focal, opt_pt, device, scaler,
                            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                            contrastive_weight=args.contrastive_weight,
                        )
                        total_loss += tl; focal_loss_sum += fl_val; cl_sum += cl_val
                        n_batches += 1
                        if bi % 500 == 0 and bi > 0:
                            print(
                                f"    Ep {ep:02d} batch {bi:5d}/{len(pretrain_loader)} "
                                f"loss={total_loss/n_batches:.4f}",
                                flush=True,
                            )
                except KeyboardInterrupt:
                    save_resume("pretrain", ep, optimizer=opt_pt, scheduler=sched_pt)
                    print(f"\n[MC-KANT] Paused. Resume with the same command; saved: {resume_path}", flush=True)
                    raise

                sched_pt.step()
                elapsed = time.time() - t_ep
                print(
                    f"    Ep {ep:03d}/{args.epochs_pretrain} | "
                    f"Total={total_loss/n_batches:.4f} | "
                    f"Focal={focal_loss_sum/n_batches:.4f} | "
                    f"SupCon={cl_sum/n_batches:.4f} | "
                    f"LR={opt_pt.param_groups[0]['lr']:.2e} | "
                    f"time={elapsed:.0f}s",
                    flush=True,
                )
                save_resume("pretrain", ep + 1, optimizer=opt_pt, scheduler=sched_pt)


        # ---- Phase B: Sequence fine-tuning (end-to-end) -----------------
        print(f"\n  [Phase B] Sequence fine-tuning for {args.epochs_finetune} epochs (seq_len={args.seq_len})")

        opt_ft = torch.optim.AdamW(
            model.parameters(), lr=args.lr_finetune,
            weight_decay=args.weight_decay,
        )
        sched_ft = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_ft,
            T_0=max(1, args.epochs_finetune // args.cosine_t0_divisor),
            T_mult=args.cosine_t_mult,
            eta_min=args.lr_finetune * 1e-3,
        )

        best_val_f1 = float(active_resume.get("best_val_f1", -1.0)) if active_resume else -1.0
        best_state: Optional[Dict] = active_resume.get("best_state") if active_resume else None
        patience_count = int(active_resume.get("patience_count", 0)) if active_resume else 0
        start_finetune_epoch = 1
        if active_resume and resume_phase == "finetune":
            start_finetune_epoch = int(active_resume.get("next_epoch", 1))
            if active_resume.get("optimizer_state_dict") is not None:
                opt_ft.load_state_dict(active_resume["optimizer_state_dict"])
            if active_resume.get("scheduler_state_dict") is not None:
                sched_ft.load_state_dict(active_resume["scheduler_state_dict"])
            print(f"  Resuming Phase B at epoch {start_finetune_epoch}/{args.epochs_finetune}", flush=True)

        run_finetune = resume_phase not in ("test", "fold_complete")
        for ep in range(start_finetune_epoch, args.epochs_finetune + 1):
            if not run_finetune:
                break
            t_ep = time.time()
            try:
                tr = _run_epoch(
                    model, seq_train_loader, focal, opt_ft, device,
                    scaler=scaler, amp_enabled=amp_enabled, amp_dtype=amp_dtype, train_mode=True,
                )
                va = _run_epoch(
                    model, seq_val_loader, focal, None, device,
                    scaler=None, amp_enabled=amp_enabled, amp_dtype=amp_dtype, train_mode=False,
                )
            except KeyboardInterrupt:
                save_resume(
                    "finetune", ep, optimizer=opt_ft, scheduler=sched_ft,
                    best_val_f1_value=best_val_f1,
                    best_state_value=best_state,
                    patience_value=patience_count,
                )
                print(f"\n[MC-KANT] Paused. Resume with the same command; saved: {resume_path}", flush=True)
                raise
            sched_ft.step()
            elapsed = time.time() - t_ep
            print(
                f"    Ep {ep:03d}/{args.epochs_finetune} | "
                f"Tr loss={tr['loss']:.4f} acc={tr['acc']:.4f} f1={tr['f1_macro']:.4f} | "
                f"Va loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['f1_macro']:.4f} | "
                f"LR={opt_ft.param_groups[0]['lr']:.2e} | {elapsed:.0f}s",
                flush=True,
            )

            if va["f1_macro"] > best_val_f1:
                best_val_f1 = va["f1_macro"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            save_resume(
                "finetune", ep + 1, optimizer=opt_ft, scheduler=sched_ft,
                best_val_f1_value=best_val_f1,
                best_state_value=best_state,
                patience_value=patience_count,
            )

            if patience_count >= args.early_stop:
                print(f"    Early stopping at epoch {ep}")
                break

        save_resume(
            "test", args.epochs_finetune + 1, optimizer=opt_ft, scheduler=sched_ft,
            best_val_f1_value=best_val_f1,
            best_state_value=best_state,
            patience_value=patience_count,
        )

        # Restore best weights before evaluation
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- Test evaluation --------------------------------------------
        te = _run_epoch(
            model, seq_test_loader, focal, None, device,
            scaler=None, amp_enabled=amp_enabled, amp_dtype=amp_dtype, train_mode=False,
        )
        print(f"\n  [Fold {fold_idx+1}] Test Results")
        print(f"    Loss: {te['loss']:.4f}  Acc: {te['acc']:.4f}  Macro-F1: {te['f1_macro']:.4f}")
        print("    Classification Report (0=W, 1=N1, 2=N2, 3=N3, 4=REM):")
        print(
            classification_report(
                te["y_true"], te["y_pred"],
                target_names=["W", "N1", "N2", "N3", "REM"],
                digits=4, zero_division=0,
            )
        )
        print("    Confusion Matrix:")
        print(confusion_matrix(te["y_true"], te["y_pred"]))

        # ---- Save checkpoint -------------------------------------------
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"mc_kant_fold{fold_idx+1:02d}_best.pt")
        torch.save(
            {
                "fold": fold_idx + 1,
                "model_state_dict": _model_for_state(model).state_dict(),
                "mean": mean,
                "std": std,
                "best_val_f1": best_val_f1,
                "test_loss": te["loss"],
                "test_acc": te["acc"],
                "test_f1_macro": te["f1_macro"],
                "train_subjects": [os.path.basename(f) for f in train_files],
                "val_subjects": [os.path.basename(f) for f in val_files],
                "test_subjects": [os.path.basename(f) for f in test_files],
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"    Checkpoint saved: {ckpt_path}")

        fold_metrics.append({
            "fold": fold_idx + 1,
            "best_val_f1": best_val_f1,
            "test_loss": te["loss"],
            "test_acc": te["acc"],
            "test_f1_macro": te["f1_macro"],
        })

        _save_resume_state(
            resume_path,
            {
                "fold_idx": fold_idx + 1,
                "phase": "fold_complete",
                "next_epoch": 1,
                "model_state_dict": None,
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
                "scaler_state_dict": None,
                "best_val_f1": -1.0,
                "best_state": None,
                "patience_count": 0,
                "fold_metrics": fold_metrics,
                "args": vars(args),
                "rng_state": _rng_state(),
            },
        )
        print(f"    Resume marker advanced to fold {fold_idx + 2}.", flush=True)

    # ---- Aggregate cross-validation results -----------------------------
    print(f"\n{'='*60}")
    print(f"  {args.n_folds}-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    accs = [m["test_acc"] for m in fold_metrics]
    f1s = [m["test_f1_macro"] for m in fold_metrics]
    print(f"  Accuracy   : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro-F1   : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    summary = {
        "n_folds": args.n_folds,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "fold_metrics": fold_metrics,
    }
    summary_path = os.path.join(args.out_dir, "mc_kant_cv_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    _delete_resume_state(args)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MC-KANT: Multimodal Contrastive KAN-Transformer training"
    )

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to folder with SN_*.npz files")
    p.add_argument("--out_dir", type=str, default="./mc_kant_checkpoints")
    p.add_argument("--n_folds", type=int, default=10,
                   help="Number of patient-wise cross-validation folds")

    # Model architecture
    p.add_argument("--fusion_dim", type=int, default=256,
                   help="Per-modality feature extractor output / epoch embedding dim")
    p.add_argument("--seq_len", type=int, default=100,
                   help="Sliding window of consecutive epochs fed to U-Net")
    p.add_argument("--unet_base_filters", type=int, default=128)
    p.add_argument("--unet_levels", type=int, default=3,
                   help="U-Net encoder down-sampling levels")
    p.add_argument("--transformer_heads", type=int, default=8)
    p.add_argument("--transformer_layers", type=int, default=4)
    p.add_argument("--proj_dim", type=int, default=128,
                   help="Contrastive projection head output dim")
    p.add_argument("--kan_hidden_dims", type=int, nargs="+", default=[256, 128],
                   help="KAN hidden layer widths (space-separated list)")
    p.add_argument("--kan_grid_size", type=int, default=8,
                   help="B-spline grid knot count per KAN edge")
    p.add_argument("--dropout", type=float, default=0.2)

    # Training
    p.add_argument("--epochs_pretrain", type=int, default=30,
                   help="Contrastive pre-training epochs (0 to skip)")
    p.add_argument("--epochs_finetune", type=int, default=60,
                   help="End-to-end sequence fine-tuning epochs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_batch_size", type=int, default=0,
                   help="Fine-tuning sequence batch size. 0 = derive from batch_size / seq_len")
    p.add_argument("--lr_pretrain", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop", type=int, default=10)

    # Loss
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss focusing parameter γ")
    p.add_argument("--supcon_temp", type=float, default=0.07,
                   help="SupCon temperature τ")
    p.add_argument("--contrastive_weight", type=float, default=0.5,
                   help="Weight of contrastive loss relative to focal loss")

    # LR scheduler
    p.add_argument("--cosine_t0_divisor", type=int, default=3,
                   help="T_0 = total_epochs // cosine_t0_divisor for warm restarts")
    p.add_argument("--cosine_t_mult", type=int, default=2,
                   help="T_mult multiplier for CosineAnnealingWarmRestarts")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "off"], default="bf16",
                   help="A100-friendly mixed precision dtype")
    p.add_argument("--require_cuda", action="store_true",
                   help="Fail instead of falling back to CPU if CUDA is unavailable")
    p.add_argument("--compile_model", action="store_true",
                   help="Use torch.compile when available")
    p.add_argument("--no_tf32", dest="tf32", action="store_false",
                   help="Disable TF32 matmul/convolution on NVIDIA Ampere GPUs")
    p.set_defaults(tf32=True)
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    p.set_defaults(persistent_workers=True)
    p.add_argument("--prefetch_factor", type=int, default=2)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_kfold(args)
