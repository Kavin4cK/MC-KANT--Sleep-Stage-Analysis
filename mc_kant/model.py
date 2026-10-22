"""
MC-KANT — Full Model Assembly
==============================
Assembles all six phases into a single nn.Module:

    MCKANT(
        EpochFeatureExtractor  ×N_modalities   (Phase 2)
        ProjectionHead                          (Phase 4 – contrastive pre-training)
        UNetContextEncoder                      (Phase 3)
        KANClassificationHead                   (Phase 5)
    )

Forward modes
-------------
The model has two operational modes controlled by `contrastive_mode` arg:

  contrastive_mode=True  (pre-training phase)
    Input  : [B, C, T] single epoch
    Output : (logits [B, 5], proj_z [B, proj_dim])
             – logits from KAN head for supervision
             – proj_z for SupCon loss

  contrastive_mode=False  (fine-tuning / inference — default)
    Input  : [B, N, C, T] sequence of N epochs (sliding window)
    Output : logits [B, N, 5]  — one set of logits per epoch in the window

The two-phase training recipe is:
  Phase A: Pre-train with SupCon loss only (no KAN head) on single epochs.
  Phase B: Fine-tune end-to-end with (FocalLoss + α·SupCon) on N-epoch windows.

Tensor shapes annotated inline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import EpochFeatureExtractor
from .unet_encoder import UNetContextEncoder
from .contrastive import ProjectionHead
from .kan_head import KANClassificationHead


# ---------------------------------------------------------------------------
# Focal Loss (Phase 6 criterion — defined here for convenience)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Designed to down-weight well-classified examples (high p_t) and focus
    training on hard, misclassified examples — particularly important for
    N1/N3 stages which are severely under-represented in the iSLEEPS dataset.

    Args:
        gamma     : Focusing parameter (γ=2 standard; higher → more focus on hard)
        alpha     : Per-class weighting tensor [n_classes] or scalar.  Pass
                    the inverse-frequency weights from build_class_weights().
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, C] raw un-normalised class scores
            targets : [B]  integer class indices 0–4

        Returns:
            Focal loss scalar (or per-sample if reduction='none')
        """
        # log softmax for numerical stability (avoid log(exp(x) / Σexp))
        log_probs = F.log_softmax(logits, dim=1)           # [B, C]
        probs = torch.exp(log_probs)                        # [B, C]

        targets_unsq = targets.view(-1, 1)                  # [B, 1]
        log_pt = log_probs.gather(1, targets_unsq).squeeze(1)  # [B]
        pt = probs.gather(1, targets_unsq).squeeze(1)          # [B]

        # Compute α_t
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(logits.device).gather(0, targets)  # [B]
        elif self.alpha is not None:
            alpha_t = float(self.alpha)
        else:
            alpha_t = 1.0

        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * log_pt  # [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Full MC-KANT Model
# ---------------------------------------------------------------------------

class MCKANT(nn.Module):
    """
    Multimodal Contrastive KAN-Transformer (MC-KANT).

    Architecture summary
    --------------------
    ┌─────────────────────────────────────────────────────┐
    │           Per-Epoch Multi-Stream Encoding           │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
    │  │ EEG Extractor│  │ EOG Extractor│  │EMG Extrc.│  │
    │  │ (Dilated     │  │ (Dilated     │  │(Dilated  │  │
    │  │  Conv1D +    │  │  Conv1D +    │  │ Conv1D + │  │
    │  │  STFT CNN +  │  │  STFT CNN +  │  │ STFT CNN)│  │
    │  │  AttnFusion) │  │  AttnFusion) │  │          │  │
    │  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
    │         └──────────────┬──┘               │        │
    │                        │   Concatenate     │        │
    │                        └────────┬──────────┘        │
    │                    Modal concat + Linear projection  │
    └────────────────────────┬────────────────────────────┘
                             │ [B*N, D_fused]
                             │
              ┌──────────────▼──────────────┐
              │   Contrastive Projection     │ (training only)
              │   Head (2-layer MLP)         │
              └──────────────┬──────────────┘
                             │ z [B, proj_dim]  → SupCon loss
                             │ h [B*N, D_fused] → U-Net
                             │
              ┌──────────────▼──────────────┐
              │   1D U-Net Contextual        │
              │   Sequence Encoder           │
              │   + Transformer Bottleneck   │
              └──────────────┬──────────────┘
                             │ [B, N, D_fused]
                             │
              ┌──────────────▼──────────────┐
              │   KAN Classification Head    │
              │   (B-spline learnable edges) │
              │   → 5 AASM stage logits      │
              └─────────────────────────────┘

    Args:
        eeg_in_channels     : EEG input channels (1 for single C4 lead)
        eog_in_channels     : EOG channels (1 for E1:M2)
        emg_in_channels     : EMG channels (1 for Chin EMG; 0 to disable)
        fusion_dim          : Per-modality feature extractor output dimension
        seq_len             : Number of consecutive epochs per training sample
        unet_base_filters   : U-Net encoder base feature maps
        unet_levels         : Number of encoder down-sampling levels
        transformer_heads   : Bottleneck transformer heads
        transformer_layers  : Bottleneck transformer depth
        proj_dim            : Projection head output dimension (contrastive)
        kan_hidden_dims     : KAN head hidden layer widths
        kan_grid_size       : KAN B-spline grid knots
        kan_spline_order    : KAN B-spline polynomial degree
        n_classes           : Output classes (5 AASM stages)
        dropout             : Global dropout rate
    """

    def __init__(
        self,
        eeg_in_channels: int = 1,
        eog_in_channels: int = 1,
        emg_in_channels: int = 1,
        fusion_dim: int = 256,
        seq_len: int = 100,
        unet_base_filters: int = 128,
        unet_levels: int = 3,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        proj_dim: int = 128,
        kan_hidden_dims: List[int] = [256, 128],
        kan_grid_size: int = 8,
        kan_spline_order: int = 3,
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_classes = n_classes

        # ---- Modality feature extractors (Phase 2) ------------------------
        self.eeg_extractor = EpochFeatureExtractor(
            in_channels=eeg_in_channels,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.eog_extractor = None
        if eog_in_channels > 0:
            self.eog_extractor = EpochFeatureExtractor(
                in_channels=eog_in_channels,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )

        self.emg_extractor = None
        if emg_in_channels > 0:
            self.emg_extractor = EpochFeatureExtractor(
                in_channels=emg_in_channels,
                # Wider frequency range for EMG (10-70 Hz), use smaller STFT
                n_fft=128,
                hop_length=32,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )

        # Count active modalities to determine concatenation dim
        n_active = 1  # EEG always active
        if self.eog_extractor is not None:
            n_active += 1
        if self.emg_extractor is not None:
            n_active += 1
        concat_dim = fusion_dim * n_active

        # Linear projection after concatenation → common epoch embedding dim
        self.modal_proj = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.epoch_dim = fusion_dim

        # ---- Contrastive projection head (Phase 4) -----------------------
        self.proj_head = ProjectionHead(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim * 2,
            proj_dim=proj_dim,
        )

        # ---- U-Net Contextual Encoder (Phase 3) --------------------------
        # max_bottleneck_len should be seq_len // 2^unet_levels (ceiling)
        max_bot = max(1, (seq_len + (2 ** unet_levels) - 1) // (2 ** unet_levels))
        self.unet = UNetContextEncoder(
            epoch_dim=fusion_dim,
            n_encoder_levels=unet_levels,
            base_filters=unet_base_filters,
            n_transformer_heads=transformer_heads,
            n_transformer_layers=transformer_layers,
            max_bottleneck_len=max_bot,
            dropout=dropout,
        )

        # ---- KAN Classification Head (Phase 5) ---------------------------
        self.kan_head = KANClassificationHead(
            in_dim=fusion_dim,
            hidden_dims=kan_hidden_dims,
            n_classes=n_classes,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            dropout=dropout,
        )

    # -----------------------------------------------------------------------
    # Internal: encode a single epoch across all modalities
    # -----------------------------------------------------------------------

    def _encode_epoch(
        self,
        eeg: torch.Tensor,
        eog: Optional[torch.Tensor] = None,
        emg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract and fuse multimodal features for a single epoch.

        Args:
            eeg : [B, 1, T_eeg]
            eog : [B, 1, T_eog]  or None
            emg : [B, 1, T_emg]  or None

        Returns:
            epoch_emb : [B, fusion_dim]
        """
        feats = [self.eeg_extractor(eeg)]   # [B, fusion_dim]

        if self.eog_extractor is not None and eog is not None:
            feats.append(self.eog_extractor(eog))

        if self.emg_extractor is not None and emg is not None:
            feats.append(self.emg_extractor(emg))

        concat = torch.cat(feats, dim=-1)        # [B, fusion_dim * n_active]
        return self.modal_proj(concat)           # [B, fusion_dim]

    # -----------------------------------------------------------------------
    # Forward pass (two modes)
    # -----------------------------------------------------------------------

    def forward(
        self,
        eeg: torch.Tensor,
        eog: Optional[torch.Tensor] = None,
        emg: Optional[torch.Tensor] = None,
        contrastive_mode: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full forward pass.

        Contrastive mode (pre-training):
            eeg : [B, 1, T]    — single epoch
            Returns: (logits [B, 5], proj_z [B, proj_dim])

        Sequence mode (fine-tuning / inference):
            eeg : [B, N, 1, T] — sliding window of N epochs
            Returns: logits [B, N, 5]

        Args:
            eeg              : EEG input tensor (see above)
            eog              : Optional EOG tensor (same shape prefix)
            emg              : Optional EMG tensor (same shape prefix)
            contrastive_mode : If True, run in single-epoch contrastive mode.

        Returns:
            In contrastive mode : (logits, proj_z)
            In sequence mode   : logits [B, N, n_classes]
        """
        if contrastive_mode:
            return self._forward_contrastive(eeg, eog, emg)
        else:
            return self._forward_sequence(eeg, eog, emg)

    def _forward_contrastive(
        self,
        eeg: torch.Tensor,
        eog: Optional[torch.Tensor],
        emg: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-epoch forward for contrastive pre-training.

        Input shapes: [B, 1, T]
        Returns: (logits [B, 5], z [B, proj_dim])
        """
        # eeg: [B, 1, T_eeg]
        h = self._encode_epoch(eeg, eog, emg)   # [B, fusion_dim]

        # Projection head for SupCon loss
        z = self.proj_head(h)                    # [B, proj_dim]

        # Direct classification (no U-Net for single epoch)
        logits = self.kan_head(h)                # [B, n_classes]
        return logits, z

    def _forward_sequence(
        self,
        eeg: torch.Tensor,
        eog: Optional[torch.Tensor],
        emg: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        N-epoch sequence forward for full end-to-end training.

        Input shapes: [B, N, 1, T]
        Returns: logits [B, N, n_classes]
        """
        # eeg: [B, N, 1, T_eeg]
        B, N, C, T = eeg.shape

        # Reshape to [B*N, C, T] for batch processing through extractor
        eeg_flat = eeg.view(B * N, C, T)
        eog_flat = eog.view(B * N, *eog.shape[2:]) if eog is not None else None
        emg_flat = emg.view(B * N, *emg.shape[2:]) if emg is not None else None

        # Per-epoch encoding
        epoch_embs = self._encode_epoch(eeg_flat, eog_flat, emg_flat)  # [B*N, D]
        epoch_embs = epoch_embs.view(B, N, -1)                          # [B, N, D]

        # U-Net contextual encoding over the N-epoch sequence
        ctx_embs = self.unet(epoch_embs)         # [B, N, D]

        # KAN classification head applied independently to each position
        # Reshape to [B*N, D] → [B*N, n_classes] → [B, N, n_classes]
        ctx_flat = ctx_embs.view(B * N, -1)      # [B*N, D]
        logits = self.kan_head(ctx_flat)          # [B*N, n_classes]
        logits = logits.view(B, N, -1)            # [B, N, n_classes]
        return logits

    # -----------------------------------------------------------------------
    # Convenience: count parameters
    # -----------------------------------------------------------------------

    def param_count(self) -> Dict[str, int]:
        """Return parameter counts by sub-module."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        counts: Dict[str, int] = {
            "eeg_extractor": _count(self.eeg_extractor),
            "proj_head": _count(self.proj_head),
            "unet": _count(self.unet),
            "kan_head": _count(self.kan_head),
        }
        if self.eog_extractor is not None:
            counts["eog_extractor"] = _count(self.eog_extractor)
        if self.emg_extractor is not None:
            counts["emg_extractor"] = _count(self.emg_extractor)

        counts["total"] = sum(counts.values())
        return counts
