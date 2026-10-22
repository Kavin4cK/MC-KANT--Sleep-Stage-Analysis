"""
Phase 2 — Dual-Stream Multi-Scale Feature Extraction
=====================================================
Three nn.Module components:

  TemporalBranch         – 1-D dilated convolutions with exponentially
                           increasing dilation rates to capture sleep rhythms
                           from spindle-level (0.5 s) to arousal-level (>10 s).

  SpectralBranch         – STFT of the signal → 2-D CNN on the spectrogram to
                           extract frequency-band power dynamics (delta, theta,
                           alpha, sigma, beta).

  AttentionFeatureFusion – Concatenates temporal + spectral streams, then
                           applies a multi-head self-attention layer to
                           dynamically re-weight the fused representation and
                           suppress motion / EMG artifacts.

Tensor shapes are annotated inline for each forward pass.
All modules receive a single input channel x of shape [B, 1, T].
The final fused representation has shape [B, fusion_dim].

Typical EEG input: B=batch, 1 channel, T=3000 (100 Hz × 30 s)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: squeeze-excitation recalibration block
# ---------------------------------------------------------------------------

class _SEBlock(nn.Module):
    """Channel squeeze-and-excitation recalibration (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        s = self.pool(x).squeeze(-1)   # [B, C]
        s = self.fc(s).unsqueeze(-1)   # [B, C, 1]
        return x * s                   # [B, C, T]


# ---------------------------------------------------------------------------
# Phase 2A – Temporal Branch (dilated 1-D convolutions)
# ---------------------------------------------------------------------------

class TemporalBranch(nn.Module):
    """
    Multi-scale temporal feature extractor using 1-D dilated convolutions.

    Dilation schedule: [1, 2, 4, 8, 16] — each doubles the receptive field.
    With kernel_size=7 and 100 Hz data:
        d=1  → receptive field  0.07 s  (spindle fine structure)
        d=2  → 0.14 s
        d=4  → 0.28 s
        d=8  → 0.56 s  (sleep spindle ~ 0.5–3 s)
        d=16 → 1.12 s  (K-complex ~ 0.5–1.5 s)

    Outputs are pooled to a fixed-length vector via adaptive average pooling.

    Args:
        in_channels  : Number of input signal channels (1 for single EEG)
        base_filters : Feature maps in the first convolution (doubled each block)
        dilation_rates: List of dilation factors (exponential growth recommended)
        pool_size    : Global adaptive pool output length
        dropout      : Dropout probability after each dilated block
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        dilation_rates: List[int] = [1, 2, 4, 8, 16],
        pool_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Initial stem: map raw signal to base_filters feature maps
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.SiLU(),
            nn.MaxPool1d(2),  # [B, F, T/2]
        )

        # Dilated blocks — each independently extracts a different temporal scale
        layers: List[nn.Module] = []
        in_ch = base_filters
        for i, d in enumerate(dilation_rates):
            # Cap max filters at 256 to control parameter count
            out_ch = min(base_filters * (2 ** min(i, 3)), 256)
            layers.append(
                self._make_dilated_block(in_ch, out_ch, dilation=d, dropout=dropout)
            )
            in_ch = out_ch

        self.dilated_blocks = nn.ModuleList(layers)
        self.final_ch = in_ch  # output channels after last block

        # Aggregate along time → [B, final_ch, pool_size]
        self.pool = nn.AdaptiveAvgPool1d(pool_size)

        # Linear projection to fixed embedding dim for fusion
        self.embed_dim = self.final_ch * pool_size
        # We'll flatten; downstream fusion projects further

    @staticmethod
    def _make_dilated_block(
        in_ch: int, out_ch: int, dilation: int, dropout: float
    ) -> nn.Sequential:
        """
        Residual-style dilated Conv1d block.
        Input : [B, in_ch, T]
        Output: [B, out_ch, T]  (T unchanged — padding=dilation*(k-1)//2)
        """
        padding = dilation * 3  # kernel_size=7 → (k-1)/2 * dilation = 3*d
        block = nn.Sequential(
            # Depthwise dilated conv (efficient, large effective receptive field)
            nn.Conv1d(
                in_ch, in_ch,
                kernel_size=7, dilation=dilation,
                padding=padding, groups=in_ch, bias=False,
            ),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            _SEBlock(out_ch),
            nn.Dropout(dropout),
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_channels, T]   e.g. [B, 1, 3000]

        Returns:
            feat : [B, embed_dim]   where embed_dim = final_ch * pool_size
        """
        # x : [B, 1, 3000]
        h = self.stem(x)              # [B, base_filters, T/2]

        # Pass through each dilated block sequentially
        for block in self.dilated_blocks:
            h = block(h)              # [B, final_ch, T'] – T' shrinks due to striding in stem

        h = self.pool(h)              # [B, final_ch, pool_size]
        feat = h.flatten(1)           # [B, final_ch * pool_size]
        return feat                   # [B, embed_dim]


# ---------------------------------------------------------------------------
# Phase 2B – Spectral Branch (STFT + 2-D CNN)
# ---------------------------------------------------------------------------

class SpectralBranch(nn.Module):
    """
    Spectral feature extractor via Short-Time Fourier Transform (STFT)
    followed by a 2-D CNN that processes the log-magnitude spectrogram.

    EEG-tuned STFT parameters (for 100 Hz data):
        n_fft      = 256  → frequency resolution 100/256 ≈ 0.39 Hz/bin
        hop_length =  50  → 50% overlap, 0.5 s frames
        For 30 s epoch at 100 Hz (T=3000):
            # frames    = (3000 - 256) / 50 + 1 = 56   → rounded to 60 via padding
            # freq bins = n_fft // 2 + 1 = 129

    The 2-D CNN operates on the spectrogram shaped [B, 1, F, Frames].
    Frequency dim F captures delta/theta/alpha/sigma/beta bands.

    Args:
        n_fft       : STFT window length
        hop_length  : STFT hop (stride)
        base_filters: Base conv filters for the 2-D CNN
        pool_output : Spatial size after adaptive pool (H, W)
        dropout     : Dropout after each 2-D conv block
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 50,
        base_filters: int = 32,
        pool_output: Tuple[int, int] = (8, 8),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pool_output = pool_output

        # 2-D CNN on log-magnitude spectrogram [B, 1, F, Frames]
        freq_bins = n_fft // 2 + 1  # 129 for n_fft=256

        self.cnn2d = nn.Sequential(
            # Block 1
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.SiLU(),
            nn.MaxPool2d(2),  # [B, F, freq/2, frames/2]
            nn.Dropout2d(dropout),

            # Block 2 – deeper features for band-specific patterns
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.SiLU(),
            nn.MaxPool2d(2),  # [B, 2F, freq/4, frames/4]
            nn.Dropout2d(dropout),

            # Block 3 – capture spectrotemporal dynamics
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.SiLU(),
            nn.Dropout2d(dropout),
        )

        # Adaptive pool → fixed spatial size regardless of input length
        self.pool = nn.AdaptiveAvgPool2d(pool_output)
        self.embed_dim = base_filters * 4 * pool_output[0] * pool_output[1]

    def _compute_stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-magnitude STFT spectrogram for multi-channel input.

        For multi-channel input [B, C, T], compute per-channel spectrograms
        and average across channels → mean power spectrum [B, 1, F, frames].

        Args:
            x : [B, C, T]  (C >= 1)

        Returns:
            log_mag : [B, 1, freq_bins, n_frames]
        """
        B, C, T = x.shape

        # Flatten batch × channels for batched STFT
        x_flat = x.reshape(B * C, T)   # [B*C, T]

        window = torch.hann_window(self.n_fft, device=x.device)

        stft_out = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # [B*C, freq_bins, n_frames]

        mag = stft_out.abs() + 1e-8          # [B*C, F, frames]
        log_mag = torch.log(mag)             # log-magnitude

        # Reshape back and average across channels → mean power spectrum
        F_bins, n_frames = log_mag.shape[1], log_mag.shape[2]
        log_mag = log_mag.view(B, C, F_bins, n_frames)   # [B, C, F, frames]
        log_mag = log_mag.mean(dim=1, keepdim=True)       # [B, 1, F, frames]
        return log_mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, T]   (C = number of input channels; handles C>1)

        Returns:
            feat : [B, embed_dim]
        """
        # x: [B, C, T] — multi-channel OK; spectrogram averages across C
        spec = self._compute_stft(x)    # [B, 1, F, frames]
        h = self.cnn2d(spec)            # [B, 4*base_filters, H', W']
        h = self.pool(h)                # [B, 4*base_filters, H, W]
        feat = h.flatten(1)             # [B, embed_dim]
        return feat



# ---------------------------------------------------------------------------
# Phase 2C – Attention Feature Fusion
# ---------------------------------------------------------------------------

class AttentionFeatureFusion(nn.Module):
    """
    Fuses temporal and spectral stream embeddings via:
        1. Linear projection of each stream to a common `fusion_dim`
        2. Stack as a 2-token sequence: [temporal_token, spectral_token]
        3. Multi-head self-attention (MHSA) to learn cross-modal dependencies
        4. Weighted sum → final `fusion_dim` embedding

    Motion artifacts and EMG contamination tend to excite high-frequency
    spectral bins; MHSA can learn to down-weight those tokens.

    Args:
        temporal_dim : Output dimensionality of TemporalBranch
        spectral_dim : Output dimensionality of SpectralBranch
        fusion_dim   : Common projection / output dimension
        n_heads      : Number of attention heads (fusion_dim must be divisible)
        dropout      : Attention dropout
    """

    def __init__(
        self,
        temporal_dim: int,
        spectral_dim: int,
        fusion_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project each stream to common space
        self.proj_temp = nn.Sequential(
            nn.Linear(temporal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.proj_spec = nn.Sequential(
            nn.Linear(spectral_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Learnable modality embedding (positional-style encoding)
        self.modal_embed = nn.Parameter(torch.zeros(1, 2, fusion_dim))
        nn.init.trunc_normal_(self.modal_embed, std=0.02)

        # Multi-head self-attention over the 2-token sequence
        self.mhsa = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.drop = nn.Dropout(dropout)

        # Final MLP to blend attended tokens
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.out_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        temp_feat: torch.Tensor,
        spec_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temp_feat : [B, temporal_dim]
            spec_feat : [B, spectral_dim]

        Returns:
            fused : [B, fusion_dim]
        """
        # Project both streams to [B, fusion_dim]
        t = self.proj_temp(temp_feat)   # [B, fusion_dim]
        s = self.proj_spec(spec_feat)   # [B, fusion_dim]

        # Stack as 2-token sequence: [B, 2, fusion_dim]
        tokens = torch.stack([t, s], dim=1)   # [B, 2, fusion_dim]
        tokens = tokens + self.modal_embed     # add modality-specific bias

        # Self-attention over 2 tokens
        attended, _ = self.mhsa(tokens, tokens, tokens)  # [B, 2, fusion_dim]
        attended = self.norm(tokens + self.drop(attended))  # residual + LN

        # Flatten 2 tokens and pass through FFN
        flat = attended.reshape(attended.shape[0], -1)   # [B, 2 * fusion_dim]
        fused = self.ffn(flat)                           # [B, fusion_dim]
        fused = self.out_norm(fused)                     # [B, fusion_dim]
        return fused


# ---------------------------------------------------------------------------
# Assembled per-epoch feature extractor
# ---------------------------------------------------------------------------

class EpochFeatureExtractor(nn.Module):
    """
    Combines TemporalBranch + SpectralBranch + AttentionFeatureFusion
    for a single modality (typically EEG).

    If you want to run separate extractors for EEG / EOG / EMG, instantiate
    three copies of this module with the same or different hyperparameters.

    Args:
        in_channels    : Signal channels in input (1 for single EEG lead)
        temporal_pool  : TemporalBranch adaptive pool size
        spectral_pool  : SpectralBranch 2-D pool spatial size
        fusion_dim     : AttentionFeatureFusion output dim
        n_heads        : Attention heads in fusion
        dropout        : Dropout probability (shared across sub-modules)
    """

    def __init__(
        self,
        in_channels: int = 1,
        temporal_pool: int = 64,
        base_temporal_filters: int = 32,
        dilation_rates: List[int] = [1, 2, 4, 8, 16],
        n_fft: int = 256,
        hop_length: int = 50,
        spectral_pool: Tuple[int, int] = (8, 8),
        base_spectral_filters: int = 32,
        fusion_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.temporal = TemporalBranch(
            in_channels=in_channels,
            base_filters=base_temporal_filters,
            dilation_rates=dilation_rates,
            pool_size=temporal_pool,
            dropout=dropout,
        )
        self.spectral = SpectralBranch(
            n_fft=n_fft,
            hop_length=hop_length,
            base_filters=base_spectral_filters,
            pool_output=spectral_pool,
            dropout=dropout,
        )
        self.fusion = AttentionFeatureFusion(
            temporal_dim=self.temporal.embed_dim,
            spectral_dim=self.spectral.embed_dim,
            fusion_dim=fusion_dim,
            n_heads=n_heads,
            dropout=dropout / 2,
        )
        self.out_dim = fusion_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_channels, T]

        Returns:
            [B, fusion_dim]
        """
        t_feat = self.temporal(x)   # [B, temporal_embed_dim]
        s_feat = self.spectral(x)   # [B, spectral_embed_dim]
        fused = self.fusion(t_feat, s_feat)   # [B, fusion_dim]
        return fused
