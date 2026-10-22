"""
MC-KANT A100 training launcher.

This file is intentionally self-contained enough for a fresh GPU server:
it can install missing Python packages, then launch patient-wise MC-KANT
training against the preprocessed iSLEEPS numpy subject files.

Typical A100 run:
    python train_mc_kant.py --install-torch

If the CUDA PyTorch wheel is already installed:
    python train_mc_kant.py --skip-install
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


DATA_DIR = "/media/rvcse22/CSERV/Sleep_CK_26/processed_data/numpy_subjects"
OUT_DIR = "./mc_kant_checkpoints_a100"

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

PYPI_PACKAGES = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "tqdm>=4.65",
    "pandas>=2.0",
    "openpyxl>=3.1",
    "mne>=1.4",
]

IMPORT_CHECKS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "mne": "mne",
}


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _run_pip(packages: list[str], extra_args: list[str] | None = None) -> None:
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    if extra_args:
        cmd.extend(extra_args)
    print("[MC-KANT] Installing:", " ".join(packages), flush=True)
    subprocess.check_call(cmd)


def bootstrap_dependencies(args: argparse.Namespace) -> None:
    """Install missing runtime dependencies before importing project modules."""
    if args.skip_install or os.environ.get("MC_KANT_SKIP_INSTALL") == "1":
        return

    pip_extra_args = args.pip_extra_arg or []

    if args.install_torch or not _has_module("torch"):
        torch_cmd = [
            "torch>=2.1",
            "torchvision>=0.16",
            "--index-url",
            args.torch_index_url,
        ]
        _run_pip(torch_cmd, pip_extra_args)

    missing = [
        package
        for module_name, package in IMPORT_CHECKS.items()
        if not _has_module(module_name)
    ]
    _run_pip(missing, pip_extra_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-bootstrapping MC-KANT A100 training launcher"
    )

    # Bootstrap/install controls. These are handled before importing torch.
    parser.add_argument("--skip-install", action="store_true",
                        help="Do not install missing packages")
    parser.add_argument("--install-torch", action="store_true",
                        help="Install/reinstall CUDA PyTorch from the configured wheel index")
    parser.add_argument("--torch-index-url", default=TORCH_INDEX_URL,
                        help="PyTorch CUDA wheel index URL")
    parser.add_argument("--pip-extra-arg", action="append",
                        help="Extra argument passed to pip install; can be repeated")

    # Data and output.
    parser.add_argument("--data_dir", default=DATA_DIR,
                        help="Folder containing SN_*.npz preprocessed subjects")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Checkpoint/metrics output directory")
    parser.add_argument("--n_folds", type=int, default=10)

    # A100-friendly model settings. Keep the model compact, scale throughput.
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--unet_base_filters", type=int, default=64)
    parser.add_argument("--unet_levels", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--kan_hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Training.
    parser.add_argument("--epochs_pretrain", type=int, default=20)
    parser.add_argument("--epochs_finetune", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Single-epoch pretraining batch size")
    parser.add_argument("--seq_batch_size", type=int, default=32,
                        help="Sequence fine-tuning batch size")
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=12)

    # Loss and scheduler.
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--supcon_temp", type=float, default=0.07)
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--cosine_t0_divisor", type=int, default=3)
    parser.add_argument("--cosine_t_mult", type=int, default=2)

    # GPU/runtime.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--compile_model", action="store_true",
                        help="Enable torch.compile")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Allow CPU fallback. By default this launcher requires CUDA.")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.set_defaults(tf32=True)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_dependencies(args)

    from mc_kant.train import train_kfold

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. "
            "Pass --data_dir if the A100 mount path is different."
        )

    args.require_cuda = not args.allow_cpu

    print("=" * 72)
    print("  MC-KANT A100 Training")
    print(f"  data_dir: {args.data_dir}")
    print(f"  out_dir : {args.out_dir}")
    print("=" * 72)
    train_kfold(args)


if __name__ == "__main__":
    main()
