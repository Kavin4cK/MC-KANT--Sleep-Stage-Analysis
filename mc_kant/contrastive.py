"""
Phase 4 — Supervised Contrastive Representation Engineering
============================================================
Provides:

  ProjectionHead          – 2-layer MLP to map epoch embeddings into the
                            contrastive latent space (following SimCLR).

  augment_epoch           – Generates two augmented views of a sleep epoch via:
                              1. Channel permutation (swaps EEG channels)
                              2. Frequency masking (zeros random STFT bins)
                            These augmentations preserve sleep-discriminative
                            features while adding useful variance for contrastive
                            pre-training.

  SupervisedContrastiveLoss – SupCon loss (Khosla et al., NeurIPS 2020).
                              Pulls representations of the same AASM stage
                              together and pushes different stages apart.

Reference:
  Khosla P. et al. "Supervised Contrastive Learning." NeurIPS 2020.
  https://arxiv.org/abs/2004.11362

Loss formulation:
    L_SupCon = Σ_{i∈I} (-1/|P(i)|) Σ_{p∈P(i)} log[
                 exp(z_i · z_p / τ) /
                 Σ_{a∈A(i)} exp(z_i · z_a / τ)
               ]
where P(i) is the set of positives (same label as i, excluding i itself),
A(i) is all other anchors in the batch.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 4A – Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP that projects epoch embeddings into the normalised contrastive
    latent space.  Placed after the temporal feature extractor; discarded at
    inference time (the upstream embedding is used for classification).

    Architecture: D → hidden_dim (BN + ReLU) → proj_dim → L2-normalise

    Args:
        input_dim  : Embedding dimension from the feature extractor
        hidden_dim : Intermediate MLP width
        proj_dim   : Final projection dimension (contrastive latent space)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, input_dim]

        Returns:
            z : [B, proj_dim]  L2-normalised (unit sphere)
        """
        z = self.mlp(x)                    # [B, proj_dim]
        z = F.normalize(z, dim=1, p=2)    # unit sphere normalisation
        return z


# ---------------------------------------------------------------------------
# Phase 4B – Augmentation Functions
# ---------------------------------------------------------------------------

def augment_epoch(
    x: torch.Tensor,
    freq_mask_ratio: float = 0.15,
    channel_perm: bool = True,
    time_mask_ratio: float = 0.10,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """
    Generate an augmented view of a sleep epoch without altering class-
    discriminative low-frequency EEG morphology.

    Augmentations applied (each independently with probability 0.5):
        1. Gaussian noise injection  – mimics electrode impedance variation
        2. Frequency masking         – zeros random frequency bins in STFT
           (inspired by SpecAugment; preserves temporal waveform integrity)
        3. Channel permutation       – swaps input channels (valid when
           multiple EEG/EOG channels are present; no-op for single channel)
        4. Time masking              – zeros a contiguous temporal segment
           (simulates short-duration signal loss / movement artefact)

    Args:
        x               : [C, T] single epoch tensor (may be on CUDA)
        freq_mask_ratio : Fraction of STFT frequency bins to zero
        channel_perm    : Whether to apply channel permutation
        time_mask_ratio : Fraction of temporal samples to mask
        noise_std       : Std of additive Gaussian noise (relative to signal)

    Returns:
        x_aug : [C, T]  augmented epoch (same shape as input)
    """
    x_aug = x.clone()
    C, T = x_aug.shape

    # ---- 1. Additive Gaussian noise -------------------------------------
    if random.random() < 0.5:
        signal_std = x_aug.std().clamp(min=1e-6)
        noise = torch.randn_like(x_aug) * (noise_std * signal_std)
        x_aug = x_aug + noise

    # ---- 2. Frequency masking (applied in Fourier domain) ---------------
    if random.random() < 0.5 and freq_mask_ratio > 0:
        # Compute FFT along time axis
        X_fft = torch.fft.rfft(x_aug, dim=-1)          # [C, T//2+1]
        n_freqs = X_fft.shape[-1]
        mask_len = max(1, int(freq_mask_ratio * n_freqs))
        start = random.randint(0, n_freqs - mask_len)
        X_fft[:, start : start + mask_len] = 0.0
        x_aug = torch.fft.irfft(X_fft, n=T, dim=-1)   # [C, T]

    # ---- 3. Channel permutation ------------------------------------------
    if channel_perm and C > 1 and random.random() < 0.5:
        perm = torch.randperm(C, device=x_aug.device)
        x_aug = x_aug[perm, :]

    # ---- 4. Time masking ------------------------------------------------
    if random.random() < 0.5 and time_mask_ratio > 0:
        mask_len = max(1, int(time_mask_ratio * T))
        start = random.randint(0, T - mask_len)
        x_aug[:, start : start + mask_len] = 0.0

    return x_aug


def generate_two_views(
    x: torch.Tensor,
    **aug_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Produce two independently augmented views of a batch of epochs.

    Args:
        x : [B, C, T]

    Returns:
        view1, view2 : each [B, C, T]
    """
    B, C, T = x.shape
    view1 = torch.stack([augment_epoch(x[i], **aug_kwargs) for i in range(B)])
    view2 = torch.stack([augment_epoch(x[i], **aug_kwargs) for i in range(B)])
    return view1, view2


# ---------------------------------------------------------------------------
# Phase 4C – Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).

    Compared with vanilla contrastive loss, SupCon leverages all samples of
    the same class as positives, leading to tighter intra-class clusters and
    larger inter-class margins — particularly beneficial for the small and
    imbalanced N1/N3 sleep stage clusters.

    Args:
        temperature : Softmax temperature τ (smaller → sharper distribution)
        base_temperature : Normalisation temperature (set equal to `temperature`
                           in the original paper)
        contrast_mode : 'all' uses all views as anchors; 'one' uses only
                        the first view as the anchor.

    Input:
        features : [B, n_views, proj_dim]  L2-normalised projection vectors.
                   For each sample i, n_views augmented views are provided.
        labels   : [B]  integer class labels (0–4 AASM stages)

    Example usage:
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        z1 = proj_head(feat1)   # [B, proj_dim]
        z2 = proj_head(feat2)   # [B, proj_dim]
        features = torch.stack([z1, z2], dim=1)  # [B, 2, proj_dim]
        loss = loss_fn(features, labels)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features : [B, n_views, dim]  (L2-normalised)
            labels   : [B]  integer labels (used to build positive mask)
            mask     : Optional pre-computed [B, B] binary positive mask.
                       If both `labels` and `mask` are provided, `mask` wins.

        Returns:
            loss : scalar
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "features must be [B, n_views, dim]; "
                f"got shape {tuple(features.shape)}"
            )

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is not None and mask is None:
            # Build [B, B] mask: mask[i,j] = 1 iff labels[i] == labels[j]
            labels = labels.contiguous().view(-1, 1)  # [B, 1]
            mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
        elif mask is None:
            # Unsupervised fallback: only diagonal positives (SimCLR-style)
            mask = torch.eye(batch_size, dtype=torch.float, device=device)

        # Reshape features: [B, n_views, dim] → [B*n_views, dim]
        contrast_count = n_views
        contrast_feature = features.view(batch_size * n_views, -1)  # [B*V, dim]

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]       # [B, dim]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature     # [B*V, dim]
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute cosine similarity matrix (features already L2-normalised)
        # anchor × contrast → [B*V, B*V]
        sim = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )  # [B*anchor_count, B*contrast_count]

        # For numerical stability subtract max per row
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Expand mask from [B, B] → [B*anchor_count, B*contrast_count]
        mask = mask.repeat(anchor_count, contrast_count)  # [B*V, B*V]

        # Exclude self-contrast on the diagonal (a sample should not be its
        # own positive because the numerator would trivially be maximised)
        logits_mask = torch.ones_like(mask) - torch.eye(
            batch_size * anchor_count, device=device
        ).repeat_interleave(1, dim=0)[:mask.shape[0], :mask.shape[1]]

        # Zero out self-contrast pairs in both mask and exp
        mask = mask * logits_mask

        # Compute log-softmax denominator (sum over all negatives + positives)
        exp_sim = torch.exp(sim) * logits_mask     # [B*V, B*V]
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean log-likelihood over positives
        # Clamp to avoid dividing by 0 when a sample has no positives in batch
        n_positives = mask.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / n_positives

        # SupCon loss: negative mean (we maximise similarity → minimise loss)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
