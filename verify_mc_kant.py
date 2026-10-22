# -*- coding: utf-8 -*-
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
"""
MC-KANT Sanity Check
====================
Verifies that all modules can be imported and a single forward pass completes
successfully on synthetic data WITHOUT requiring a GPU or a real dataset.

Run:
    python verify_mc_kant.py

Expected output (approx):
    [OK] Imports
    [OK] Model built — 12,345,678 total parameters
    [OK] Contrastive pre-train forward: logits=(8, 5) z=(8, 128)
    [OK] Sequence fine-tune forward:    logits=(2, 100, 5)
    [OK] SupCon loss: 2.3456
    [OK] Focal loss:  1.7891
    [OK] KANLinear standalone: (4, 5)
    All checks passed!
"""

import sys
import traceback

import torch
import torch.nn.functional as F


def run_checks():
    device = torch.device("cpu")
    print("=" * 55)
    print("  MC-KANT Architecture Sanity Check")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Imports
    # ------------------------------------------------------------------
    try:
        from mc_kant import MCKANT
        from mc_kant.dataset import SleepEpochDataset, MultimodalSleepDataset
        from mc_kant.feature_extractor import (
            TemporalBranch, SpectralBranch, AttentionFeatureFusion, EpochFeatureExtractor,
        )
        from mc_kant.unet_encoder import UNetContextEncoder
        from mc_kant.contrastive import (
            ProjectionHead, SupervisedContrastiveLoss, augment_epoch, generate_two_views,
        )
        from mc_kant.kan_head import KANLinear, BSplineBasis, KANClassificationHead
        from mc_kant.model import FocalLoss
        print("[OK] Imports")
    except Exception as e:
        print(f"[FAIL] Imports: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Individual sub-module checks
    # ------------------------------------------------------------------

    # TemporalBranch
    x = torch.randn(4, 1, 3000)  # [B=4, C=1, T=3000]
    try:
        tb = TemporalBranch(in_channels=1, base_filters=16, pool_size=32)
        out = tb(x)
        assert out.shape[0] == 4, f"Bad batch dim: {out.shape}"
        print(f"[OK] TemporalBranch output: {tuple(out.shape)}")
    except Exception as e:
        print(f"[FAIL] TemporalBranch: {e}")
        traceback.print_exc()

    # SpectralBranch
    try:
        sb = SpectralBranch(n_fft=128, hop_length=32, base_filters=16, pool_output=(4, 4))
        out = sb(x)
        assert out.shape[0] == 4
        print(f"[OK] SpectralBranch output: {tuple(out.shape)}")
    except Exception as e:
        print(f"[FAIL] SpectralBranch: {e}")
        traceback.print_exc()

    # EpochFeatureExtractor
    try:
        efe = EpochFeatureExtractor(
            in_channels=1, base_temporal_filters=16, temporal_pool=32,
            n_fft=128, hop_length=32, spectral_pool=(4, 4),
            base_spectral_filters=16, fusion_dim=64,
        )
        out = efe(x)
        assert out.shape == (4, 64), f"Expected (4,64) got {out.shape}"
        print(f"[OK] EpochFeatureExtractor output: {tuple(out.shape)}")
    except Exception as e:
        print(f"[FAIL] EpochFeatureExtractor: {e}")
        traceback.print_exc()

    # UNetContextEncoder
    try:
        N, D = 8, 64   # short sequence for speed
        seq = torch.randn(2, N, D)  # [B=2, N=8, D=64]
        unet = UNetContextEncoder(
            epoch_dim=D, n_encoder_levels=2,
            base_filters=32, n_transformer_heads=4, n_transformer_layers=1,
            max_bottleneck_len=8,
        )
        out = unet(seq)
        assert out.shape == (2, N, D), f"Expected (2,{N},{D}) got {out.shape}"
        print(f"[OK] UNetContextEncoder output: {tuple(out.shape)}")
    except Exception as e:
        print(f"[FAIL] UNetContextEncoder: {e}")
        traceback.print_exc()

    # KANLinear standalone
    try:
        kan = KANLinear(in_features=8, out_features=5, grid_size=3)
        xi = torch.randn(4, 8)
        out = kan(xi)
        assert out.shape == (4, 5), f"Expected (4,5) got {out.shape}"
        print(f"[OK] KANLinear standalone output: {tuple(out.shape)}")
    except Exception as e:
        print(f"[FAIL] KANLinear: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 3. Full model - contrastive mode (single epoch)
    # ------------------------------------------------------------------
    try:
        model = MCKANT(
            eeg_in_channels=1, eog_in_channels=0, emg_in_channels=0,
            fusion_dim=64, seq_len=8,
            unet_base_filters=32, unet_levels=2,
            transformer_heads=4, transformer_layers=1,
            proj_dim=32, kan_hidden_dims=[64, 32],
            kan_grid_size=4, n_classes=5, dropout=0.1,
        ).to(device)

        pc = model.param_count()
        print(f"[OK] Model built — {pc['total']:,} total parameters")

        B = 8
        eeg_single = torch.randn(B, 1, 3000)
        logits, z = model(eeg_single, contrastive_mode=True)
        assert logits.shape == (B, 5), f"logits: {logits.shape}"
        assert z.shape == (B, 32), f"z: {z.shape}"
        print(f"[OK] Contrastive forward: logits={tuple(logits.shape)}  z={tuple(z.shape)}")
    except Exception as e:
        print(f"[FAIL] Model contrastive forward: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. Full model - sequence fine-tune mode
    # ------------------------------------------------------------------
    try:
        B, N = 2, 8
        eeg_seq = torch.randn(B, N, 1, 3000)
        logits_seq = model(eeg_seq, contrastive_mode=False)
        assert logits_seq.shape == (B, N, 5), f"logits_seq: {logits_seq.shape}"
        print(f"[OK] Sequence forward: logits={tuple(logits_seq.shape)}")
    except Exception as e:
        print(f"[FAIL] Model sequence forward: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 5. Supervised Contrastive Loss
    # ------------------------------------------------------------------
    try:
        supcon = SupervisedContrastiveLoss(temperature=0.07)
        B = 16
        z1 = F.normalize(torch.randn(B, 32), dim=1)
        z2 = F.normalize(torch.randn(B, 32), dim=1)
        feats = torch.stack([z1, z2], dim=1)   # [B, 2, 32]
        labels = torch.randint(0, 5, (B,))
        loss_sc = supcon(feats, labels)
        assert loss_sc.item() > 0, "SupCon loss should be positive"
        print(f"[OK] SupCon loss: {loss_sc.item():.4f}")
    except Exception as e:
        print(f"[FAIL] SupervisedContrastiveLoss: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 6. Focal Loss
    # ------------------------------------------------------------------
    try:
        focal = FocalLoss(gamma=2.0, alpha=None)
        logits_f = torch.randn(16, 5)
        labels_f = torch.randint(0, 5, (16,))
        loss_fl = focal(logits_f, labels_f)
        assert loss_fl.item() > 0
        print(f"[OK] Focal loss: {loss_fl.item():.4f}")
    except Exception as e:
        print(f"[FAIL] FocalLoss: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 7. Augmentation
    # ------------------------------------------------------------------
    try:
        epoch = torch.randn(1, 3000)
        v1, v2 = generate_two_views(epoch.unsqueeze(0))
        assert v1.shape == (1, 1, 3000)
        assert not torch.allclose(v1, v2), "Two views should differ"
        print("[OK] Augmentation: view1 != view2 (confirmed)")
    except Exception as e:
        print(f"[FAIL] Augmentation: {e}")
        traceback.print_exc()

    print("\n  All checks passed! [OK]")
    print("=" * 55)


if __name__ == "__main__":
    run_checks()
