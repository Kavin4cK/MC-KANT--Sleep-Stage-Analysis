"""
Phase 3 — U-Net Contextual Sequence Encoder
============================================
Models temporal dependencies across a sliding window of N consecutive epochs
(default N=100, representing ~50 minutes of sleep) using a 1-D Convolutional
U-Net with a Transformer bottleneck.

Architecture overview
---------------------

                         Input: [B, N, D]
                              │
            ┌─────────────────▼──────────────────┐
            │         U-Net Encoder               │
            │  Conv1d stride-2 blocks             │
            │  E1→[B, N/2,  2D]                  │
            │  E2→[B, N/4,  4D]                  │
            │  E3→[B, N/8,  8D]                  │
            └─────────────────┬──────────────────┘
                              │ skip connections
            ┌─────────────────▼──────────────────┐
            │         Bottleneck Transformer       │
            │  Multi-Head Self-Attention on        │
            │  the N/8 compressed representations │
            │  [B, N/8, 8D]  →  [B, N/8, 8D]     │
            └─────────────────┬──────────────────┘
                              │ skip connections
            ┌─────────────────▼──────────────────┐
            │         U-Net Decoder               │
            │  ConvTranspose1d up-sample blocks   │
            │  D3→[B, N/4,  4D]                  │
            │  D2→[B, N/2,  2D]                  │
            │  D1→[B, N,     D]                  │
            └─────────────────┬──────────────────┘
                              │
                         Output: [B, N, D]

Each output position corresponds to one sleep epoch; the full sequence is
used by downstream components (contrastive loss, KAN classification head).

Tensor shapes are annotated inline.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper building blocks
# ---------------------------------------------------------------------------

class _Conv1dBlock(nn.Module):
    """
    [B, C_in, L] → Conv1d → BN → GELU → [B, C_out, L']
    If stride > 1 the sequence length is halved (down-sampling encoder step).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        # Residual shortcut
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class _UpBlock(nn.Module):
    """
    Decoder up-sample block.
    Concatenates skip connection then applies Conv1dBlock.
    [B, C_in, L] + skip[B, C_skip, 2L]  →  [B, C_out, 2L]
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = _Conv1dBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : [B, in_ch, L]
            skip : [B, skip_ch, 2L] (or close — trimmed to match)

        Returns:
            [B, out_ch, 2L]
        """
        x = self.up(x)   # [B, in_ch, 2L]

        # Trim skip in case of odd-length mismatch
        if skip.shape[-1] != x.shape[-1]:
            x = x[:, :, : skip.shape[-1]]

        x = torch.cat([x, skip], dim=1)   # [B, in_ch + skip_ch, 2L]
        return self.conv(x)               # [B, out_ch, 2L]


# ---------------------------------------------------------------------------
# Bottleneck Transformer
# ---------------------------------------------------------------------------

class _BottleneckTransformer(nn.Module):
    """
    Standard Transformer encoder (Pre-LN formulation) applied at the U-Net
    bottleneck. Operates on the compressed sequence of length N/8 with
    'bottleneck_dim' channels.

    The self-attention mechanism here captures *global* stage-transition
    patterns across the compressed 50-minute recording — e.g., the
    characteristic cyclic alternation of NREM/REM.

    Args:
        dim        : Feature dimension at the bottleneck
        n_heads    : Number of attention heads
        n_layers   : Stacked transformer encoder layers
        ffn_mult   : FFN hidden dim multiplier (= dim * ffn_mult)
        max_seq_len: Maximum compressed sequence length (for learnable PE)
        dropout    : Attention & FFN dropout
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_mult: int = 4,
        max_seq_len: int = 64,   # N/8 for N=512 or up to 128 for N=1024
        dropout: float = 0.1,
    ):
        super().__init__()

        # Learnable positional embedding [1, max_seq_len, dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, L, dim]  (L = compressed sequence length at bottleneck)

        Returns:
            [B, L, dim]
        """
        L = x.shape[1]
        # Slice positional embedding to actual length (handles variable N)
        x = x + self.pos_embed[:, :L, :]   # [B, L, dim]
        x = self.transformer(x)             # [B, L, dim]
        return self.norm(x)


# ---------------------------------------------------------------------------
# Phase 3 – Full 1-D U-Net with Transformer Bottleneck
# ---------------------------------------------------------------------------

class UNetContextEncoder(nn.Module):
    """
    1-D Convolutional U-Net with a Multi-Head Self-Attention Transformer
    at the deepest bottleneck.

    Input  : [B, N, epoch_dim]   (B batches, N epochs, epoch_dim per epoch)
    Output : [B, N, epoch_dim]   (same shape — contextualised representations)

    The encoder down-samples the sequence N times by 2 (3 levels = N/8),
    the Transformer attends globally over the N/8 bottleneck tokens, and the
    decoder restores the N-length sequence via skip-connected up-sampling.
    Args:
        epoch_dim          : Input/output feature dimension per epoch (D)
        n_encoder_levels   : Number of encoder down-sampling steps (default 3)
        base_filters       : Feature maps in the first encoder block (doubled each level)
        n_transformer_heads: Heads in the bottleneck transformer
        n_transformer_layers: Layers in the bottleneck transformer
        max_bottleneck_len : Max compressed length for positional embedding
        dropout            : Dropout throughout
    """

    def __init__(
        self,
        epoch_dim: int,
        n_encoder_levels: int = 3,
        base_filters: int = 128,
        n_transformer_heads: int = 8,
        n_transformer_layers: int = 4,
        max_bottleneck_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_levels = n_encoder_levels

        # ---- Encoder ------------------------------------------------------
        # Each block down-samples by 2 and doubles the channel count.
        # Channels: D → base → 2*base → 4*base (→ … → 2^k * base)
        enc_in_dims: List[int] = []    # input channels PER block (= channels of the skip tensors)
        enc_out_dims: List[int] = []   # output channels per block

        self.encoder_blocks = nn.ModuleList()
        in_ch = epoch_dim
        for level in range(n_encoder_levels):
            out_ch = base_filters * (2 ** level)
            self.encoder_blocks.append(
                _Conv1dBlock(in_ch, out_ch, kernel_size=3, stride=2, dropout=dropout)
            )
            enc_in_dims.append(in_ch)   # skip tensor has THIS many channels
            enc_out_dims.append(out_ch)
            in_ch = out_ch

        # Channel counts stored for decoder use
        self.enc_in_dims = enc_in_dims
        self.enc_out_dims = enc_out_dims

        # ---- Bottleneck Transformer ----------------------------------------
        bottleneck_dim = enc_out_dims[-1]
        self.bottleneck = _BottleneckTransformer(
            dim=bottleneck_dim,
            n_heads=n_transformer_heads,
            n_layers=n_transformer_layers,
            max_seq_len=max_bottleneck_len,
            dropout=dropout,
        )

        # ---- Decoder -------------------------------------------------------
        # Build decoder in reverse order.
        # At each decoder level i (reversed from deepest), the _UpBlock receives:
        #   - input from above:   cur_in_ch channels
        #   - skip connection:    enc_in_dims[level] channels  (saved BEFORE each encoder block)
        # and produces:          cur_out_ch channels
        self.decoder_blocks = nn.ModuleList()
        cur_in_ch = enc_out_dims[-1]   # bottleneck output channels

        for i, level in enumerate(reversed(range(n_encoder_levels))):
            skip_ch = enc_in_dims[level]           # skip was saved BEFORE the encoder block
            # Output: match the channel count feeding into the next (shallower) decoder level
            cur_out_ch = enc_in_dims[level - 1] if level > 0 else epoch_dim
            self.decoder_blocks.append(
                _UpBlock(in_ch=cur_in_ch, skip_ch=skip_ch, out_ch=cur_out_ch, dropout=dropout)
            )
            cur_in_ch = cur_out_ch   # next block's input = this block's output

        # Final 1×1 projection back to epoch_dim
        self.out_proj = nn.Sequential(
            nn.Conv1d(epoch_dim, epoch_dim, kernel_size=1, bias=False),
            nn.LayerNorm(epoch_dim),   # applied after .transpose
        )

        self.epoch_dim = epoch_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, N, epoch_dim]  — sequence of N epoch embeddings

        Returns:
            out : [B, N, epoch_dim]  — contextualised epoch embeddings
        """
        B, N, D = x.shape

        # U-Net operates on [B, C, L] convention
        h = x.transpose(1, 2)   # [B, D, N]

        # ---- Encoder with skip connections --------------------------------
        skips: List[torch.Tensor] = []
        for block in self.encoder_blocks:
            skips.append(h)          # save skip before down-sampling
            h = block(h)             # [B, out_ch, L/2]
        # h : [B, enc_out_dims[-1], N/2^n_levels]

        # ---- Bottleneck Transformer ----------------------------------------
        # Transpose to [B, L, C] for Transformer, then back
        h = h.transpose(1, 2)        # [B, L_bot, C_bot]
        h = self.bottleneck(h)       # [B, L_bot, C_bot]
        h = h.transpose(1, 2)        # [B, C_bot, L_bot]

        # ---- Decoder with skip fusion ------------------------------------
        for i, block in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]   # pop skips in reverse order
            h = block(h, skip)       # [B, out_ch, 2*L]

        # h : [B, D, N]  — restored to original sequence length
        # Final projection + layer norm
        h = self.out_proj[0](h)      # Conv1d: [B, D, N]
        h = h.transpose(1, 2)        # [B, N, D]
        h = self.out_proj[1](h)      # LayerNorm: [B, N, D]
        return h                     # [B, N, epoch_dim]
