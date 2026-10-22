"""
Phase 5 — KAN Classification Head
===================================
Implements a Kolmogorov-Arnold Network (KAN) classification head that
replaces the traditional MLP softmax head.

Background
----------
KANs (Liu et al., 2024 — "KAN: Kolmogorov-Arnold Networks") place learnable
univariate activation functions on network *edges* (weights) rather than
fixed activations on nodes.  Each edge computes a B-spline transformation:

    φ(x) = w_b · b(x) + w_s · spline(x)

where b(x) is a basis function (SiLU by default) and spline(x) is a learnable
linear combination of B-spline basis functions evaluated at x.

This gives the network the ability to learn highly non-linear, interpretable
decision boundaries — suitable for the complex spectro-temporal patterns that
differentiate AASM sleep stages.

Implementation note
-------------------
We provide a self-contained B-spline KAN layer implementation that does NOT
require the external `pykan` package, making it pip-install-free and
fully portable.

If `pykan` is installed, an alternative adapter class `PyKANHead` is also
provided that wraps the official implementation.

Classes:
  BSplineBasis        – Computes B-spline basis functions (degree k on a grid)
  KANLinear           – Single KAN layer: learnable edge activations
  KANClassificationHead – Stacked KAN layers → 5-class softmax output
  PyKANHead           – Optional pykan-backed head (requires: pip install pykan)

Reference:
  Liu Z. et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756, 2024.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# B-Spline basis computation
# ---------------------------------------------------------------------------

class BSplineBasis(nn.Module):
    """
    Computes uniform B-spline basis functions of degree `spline_order` over a
    fixed grid of `grid_size` equally-spaced knots on [grid_range[0], grid_range[1]].

    For an input x ∈ ℝ the output is a vector of length (grid_size + spline_order)
    containing the values of each basis function evaluated at x.

    Args:
        grid_size    : Number of grid intervals (more → finer resolution)
        spline_order : Degree of B-splines (3 = cubic, default in original KAN)
        grid_range   : Domain of the grid [lo, hi] (inputs are clamped)
    """

    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Register grid as a fixed buffer (not a learnable parameter)
        lo, hi = grid_range
        h = (hi - lo) / grid_size
        # Extended grid: k extra knots on each side for the B-spline basis
        grid = torch.linspace(
            lo - spline_order * h,
            hi + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )  # [G + 2k + 1]
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B-spline basis functions at input x.

        Args:
            x : [B, in_features]  (must be in grid_range for best results)

        Returns:
            bases : [B, in_features, grid_size + spline_order]
        """
        x = x.unsqueeze(-1)  # [B, in_features, 1]

        # Clamp x to grid domain
        lo, hi = self.grid_range
        x = x.clamp(lo, hi)

        # De Boor's algorithm (recursive Cox-de Boor formula)
        # Degree-0 basis (indicator functions)
        g = self.grid  # [G + 2k + 1]
        bases = (x >= g[:-1]) & (x < g[1:])  # [B, in_features, G + 2k]
        bases = bases.to(x.dtype)

        # Elevate degree from 0 to spline_order
        for k in range(1, self.spline_order + 1):
            left_num  = x - g[: -(k + 1)]
            left_den  = (g[k:-1] - g[: -(k + 1)]).clamp(min=1e-8)
            right_num = g[k + 1 :] - x
            right_den = (g[k + 1 :] - g[1:-k]).clamp(min=1e-8)
            bases = (left_num / left_den) * bases[:, :, :-1] + \
                    (right_num / right_den) * bases[:, :, 1:]
        # bases: [B, in_features, grid_size + spline_order]
        return bases.contiguous()


# ---------------------------------------------------------------------------
# KAN Linear layer
# ---------------------------------------------------------------------------

class KANLinear(nn.Module):
    """
    A single Kolmogorov-Arnold Network (KAN) linear layer.

    Each input-output edge computes:
        φ_{i→j}(x_i) = w_b_{i,j} · silu(x_i) + Σ_k c_{i,j,k} · B_k(x_i)

    where B_k(x) are B-spline basis functions and c_{i,j,k} are learnable
    spline coefficients.

    The output is a weighted sum across input features:
        y_j = Σ_i φ_{i→j}(x_i)

    This is equivalent to a learned non-linear transformation per edge
    (as opposed to a linear weight in standard MLP layers).

    Args:
        in_features   : Input feature dimension
        out_features  : Output feature dimension
        grid_size     : B-spline grid size
        spline_order  : B-spline degree
        scale_noise   : Initial spline coefficient noise scale
        scale_base    : Scale of basis (SiLU) component weights
        scale_spline  : Scale of spline component weights
        enable_standalone_scale_spline: Separate learnable scale per output
        grid_range    : Input domain for the B-spline grid
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.basis = BSplineBasis(grid_size, spline_order, grid_range)
        n_basis = grid_size + spline_order   # number of basis functions per edge

        # Learnable spline coefficients: [out, in, n_basis]
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_basis)
        )
        # Base weight (SiLU component): [out, in]
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Optional per-output scale for spline component
        if enable_standalone_scale_spline:
            self.spline_scale = nn.Parameter(torch.ones(out_features, in_features))
        else:
            self.register_buffer("spline_scale", torch.ones(out_features, in_features))

        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self._reset_parameters(scale_noise)

    def _reset_parameters(self, scale_noise: float):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, 0.0, scale_noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_features]

        Returns:
            y : [B, out_features]
        """
        # ---- Base component (SiLU) ----------------------------------------
        # y_base = x @ base_weight^T  with SiLU activation applied element-wise
        # Shape: [B, in_features] × [in_features, out_features] → [B, out_features]
        base_out = F.linear(F.silu(x), self.scale_base * self.base_weight)

        # ---- Spline component ---------------------------------------------
        # Evaluate B-spline basis for all input features
        bases = self.basis(x)  # [B, in_features, n_basis]

        # Effective spline weight = spline_weight scaled per output
        # [out, in, n_basis] * [out, in, 1] → [out, in, n_basis]
        eff_weight = self.spline_weight * (
            self.scale_spline * self.spline_scale
        ).unsqueeze(-1)  # [out, in, n_basis]

        # Contract: Σ_k c_{i,j,k} * B_k(x_i)
        # bases: [B, in, n_basis], eff_weight: [out, in, n_basis]
        # → einsum: b,i,n  × o,i,n  → b,o
        spline_out = torch.einsum("bin,oin->bo", bases, eff_weight)  # [B, out]

        return base_out + spline_out  # [B, out_features]


# ---------------------------------------------------------------------------
# KAN Classification Head
# ---------------------------------------------------------------------------

class KANClassificationHead(nn.Module):
    """
    Multi-layer KAN head replacing the traditional MLP classification head.

    Architecture:
        Input [B, in_dim]
          → KANLinear(in_dim → hidden_dims[0]) → LayerNorm → Dropout
          → KANLinear(hidden_dims[0] → hidden_dims[1]) → LayerNorm → Dropout
          → …
          → KANLinear(hidden_dims[-1] → n_classes)
          → (logits; apply softmax externally for loss computation)

    The KAN activation functions learned per-edge can reveal which latent
    features most strongly predict each AASM stage (interpretability benefit).

    Args:
        in_dim       : Input dimension from the U-Net encoder output
        hidden_dims  : List of intermediate KAN layer widths
        n_classes    : Number of output classes (5 for AASM staging)
        grid_size    : B-spline grid resolution per edge
        spline_order : B-spline polynomial degree
        dropout      : Dropout between layers
        grid_range   : Input domain for spline basis (normalise inputs first)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [256, 128],
        n_classes: int = 5,
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.3,
        grid_range: tuple = (-4.0, 4.0),
    ):
        super().__init__()

        dims = [in_dim] + hidden_dims + [n_classes]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            layers.append(
                KANLinear(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                )
            )
            if not is_last:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.Dropout(dropout))

        self.kan_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_dim]   — epoch embedding from U-Net encoder

        Returns:
            logits : [B, n_classes]
        """
        for layer in self.kan_layers:
            x = layer(x)
        return x   # raw logits; apply F.softmax / cross_entropy externally


# ---------------------------------------------------------------------------
# Optional: pykan-backed head (requires `pip install pykan`)
# ---------------------------------------------------------------------------

class PyKANHead(nn.Module):
    """
    Classification head backed by the official `pykan` library.

    Install via:  pip install pykan

    Falls back to KANClassificationHead if pykan is not available.

    Args:
        in_dim      : Input dimension
        hidden_dims : Hidden layer widths
        n_classes   : Output classes (5 AASM stages)
        grid        : KAN grid parameter (knot density)
        k           : Spline order
        device      : Torch device string
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [256, 128],
        n_classes: int = 5,
        grid: int = 8,
        k: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        try:
            from kan import KAN as _KAN  # pykan

            dims = [in_dim] + hidden_dims + [n_classes]
            self.kan = _KAN(width=dims, grid=grid, k=k, device=device)
            self._use_pykan = True
        except ImportError:
            print(
                "[MC-KANT] pykan not found – falling back to built-in BSpline KAN head. "
                "Install via: pip install pykan"
            )
            self.kan = KANClassificationHead(
                in_dim=in_dim,
                hidden_dims=hidden_dims,
                n_classes=n_classes,
                grid_size=grid,
                spline_order=k,
            )
            self._use_pykan = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_dim]

        Returns:
            logits : [B, n_classes]
        """
        return self.kan(x)
