"""Stage-2 SED / guide encoder for Morocutti-style temporal conditioning.

This module provides a lightweight, project-friendly replacement for the
auxiliary stage-2 SED model used to generate

1. frame-wise class probability maps for Time-FiLM, and
2. hidden states for latent feature injection.

Design goals
------------
- Accept the same spectrogram layout used by the current separator pipeline:
  ``[B, 1, F, T]`` or ``[B, F, T]``.
- Return frame-wise predictions with shape ``[B, C, T]``.
- Expose per-layer hidden states with shape ``[B, T, D]`` so that
  ``conditioning.py`` can later fuse and inject them.
- Stay self-contained and not depend on external pretrained packages.

This file does *not* implement AudioSep loading by itself. Pretrained loading
should be wired later from the training / module level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Stage2SEDConfig:
    """Configuration for :class:`Stage2SEDGuideEncoder`.

    Parameters
    ----------
    num_classes:
        Number of sound-event / query classes for frame-wise prediction.
    input_channels:
        Number of spectrogram channels. The current separator pipeline uses 1.
    stem_channels:
        Channel size of the 2D convolutional stem.
    hidden_dim:
        Transformer token dimension.
    num_layers:
        Number of Transformer encoder blocks.
    num_heads:
        Number of attention heads per Transformer block.
    mlp_ratio:
        Expansion ratio in the Transformer's feed-forward network.
    dropout:
        Dropout probability used in the temporal encoder and heads.
    max_time_positions:
        Maximum cached sinusoidal positional encoding length.
    use_frequency_attention_pool:
        If ``True``, learn a frequency attention map before temporal encoding.
        Otherwise, use simple mean pooling across frequency.
    temporal_conv_kernel_size:
        Kernel size of the local temporal Conv1d applied before the Transformer.
    return_all_hidden_states:
        Whether ``forward`` returns every intermediate hidden state.
    strong_activation:
        Activation for strong/weak outputs. ``sigmoid`` fits multi-label SED.
    """

    num_classes: int
    input_channels: int = 1
    stem_channels: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_time_positions: int = 4096
    use_frequency_attention_pool: bool = True
    temporal_conv_kernel_size: int = 5
    return_all_hidden_states: bool = True
    strong_activation: str = "sigmoid"


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal tokens."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0.")
        if max_len < 1:
            raise ValueError("max_len must be >= 1.")
        pe = self._build_pe(dim=dim, length=max_len)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(dim: int, length: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}.")
        t = x.shape[1]
        if t > self.pe.shape[0]:
            pe = self._build_pe(dim=x.shape[-1], length=t).to(device=x.device, dtype=x.dtype)
        else:
            pe = self.pe[:t].to(device=x.device, dtype=x.dtype)
        return x + pe.unsqueeze(0)


class FrequencyAttentionPool(nn.Module):
    """Learnable pooling over the frequency axis.

    Input shape: ``[B, C, F, T]``
    Output shape: ``[B, C, T]``
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, F, T], got {tuple(x.shape)}.")
        logits = self.score(x)  # [B, 1, F, T]
        weights = torch.softmax(logits, dim=2)
        pooled = (weights * x).sum(dim=2)  # [B, C, T]
        return {"pooled": pooled, "weights": weights}


class Stage2TransformerBlock(nn.Module):
    """Transformer block with batch-first interface and residual pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        ff_dim = int(round(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}.")
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class AttentiveClipPooling(nn.Module):
    """Attention pooling from frame embeddings to one clip embedding."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}.")
        logits = self.attn(x)  # [B, T, 1]
        weights = torch.softmax(logits, dim=1)
        pooled = (weights * x).sum(dim=1)  # [B, D]
        return {"pooled": pooled, "weights": weights}


class Stage2SEDGuideEncoder(nn.Module):
    """Auxiliary SED / guide encoder for temporal conditioning.

    Expected input
    --------------
    - ``input_spec`` with shape ``[B, 1, F, T]`` or ``[B, F, T]``.

    Main outputs
    ------------
    - ``strong_logits`` / ``strong_probabilities``: ``[B, C, T]``
    - ``weak_logits`` / ``weak_probabilities``: ``[B, C]``
    - ``frame_features``: final frame embeddings ``[B, T, D]``
    - ``hidden_states``: list of intermediate hidden states ``[B, T, D]``
    """

    VALID_ACTIVATIONS = {"sigmoid", "softmax", "identity"}

    def __init__(self, config: Optional[Stage2SEDConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = Stage2SEDConfig(**kwargs)
        self.config = config

        if self.config.num_classes < 1:
            raise ValueError("num_classes must be >= 1.")
        if self.config.strong_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported strong_activation={self.config.strong_activation!r}. "
                f"Supported: {sorted(self.VALID_ACTIVATIONS)}"
            )
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.config.temporal_conv_kernel_size < 1 or self.config.temporal_conv_kernel_size % 2 == 0:
            raise ValueError("temporal_conv_kernel_size must be a positive odd integer.")

        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_channels, self.config.stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.config.stem_channels),
            nn.GELU(),
            nn.Conv2d(self.config.stem_channels, self.config.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.config.dropout),
        )

        self.frequency_pool = (
            FrequencyAttentionPool(self.config.hidden_dim, dropout=self.config.dropout)
            if self.config.use_frequency_attention_pool
            else None
        )

        temporal_padding = self.config.temporal_conv_kernel_size // 2
        self.temporal_pre = nn.Sequential(
            nn.Conv1d(
                self.config.hidden_dim,
                self.config.hidden_dim,
                kernel_size=self.config.temporal_conv_kernel_size,
                padding=temporal_padding,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        self.position = SinusoidalPositionalEncoding(
            dim=self.config.hidden_dim,
            max_len=self.config.max_time_positions,
        )

        self.blocks = nn.ModuleList(
            [
                Stage2TransformerBlock(
                    dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_dim)

        self.strong_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )
        self.clip_pool = AttentiveClipPooling(self.config.hidden_dim, dropout=self.config.dropout)
        self.weak_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    @staticmethod
    def _validate_input_spec(input_spec: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if not torch.is_tensor(input_spec):
            raise TypeError(f"Expected torch.Tensor, got {type(input_spec)!r}.")
        if input_spec.ndim == 3:
            input_spec = input_spec.unsqueeze(1)  # [B, 1, F, T]
        if input_spec.ndim != 4:
            raise ValueError(
                "Expected spectrogram shape [B, C, F, T] or [B, F, T], "
                f"got {tuple(input_spec.shape)}."
            )
        if input_spec.shape[1] != expected_channels:
            raise ValueError(
                f"Expected input_channels={expected_channels}, got shape {tuple(input_spec.shape)}."
            )
        return input_spec

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.strong_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.config.strong_activation == "softmax":
            return torch.softmax(logits, dim=1)
        return logits

    def stem_features(self, input_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._validate_input_spec(input_spec, expected_channels=self.config.input_channels)
        stem_feature = self.stem(x)  # [B, D, F, T]

        if self.frequency_pool is not None:
            pool_out = self.frequency_pool(stem_feature)
            temporal_tokens = pool_out["pooled"]  # [B, D, T]
            frequency_attention = pool_out["weights"]
        else:
            temporal_tokens = stem_feature.mean(dim=2)  # [B, D, T]
            frequency_attention = None

        temporal_tokens = self.temporal_pre(temporal_tokens)  # [B, D, T]
        frame_features = temporal_tokens.transpose(1, 2).contiguous()  # [B, T, D]
        return {
            "input_spec": x,
            "stem_feature": stem_feature,
            "frequency_attention": frequency_attention,
            "frame_features_pre_transformer": frame_features,
        }

    def encode(self, input_spec: torch.Tensor) -> Dict[str, object]:
        stem_out = self.stem_features(input_spec)
        x = stem_out["frame_features_pre_transformer"]
        x = self.position(x)

        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)

        frame_features = self.final_norm(x)
        strong_logits_bt = self.strong_head(frame_features)  # [B, T, C]
        strong_logits = strong_logits_bt.transpose(1, 2).contiguous()  # [B, C, T]
        strong_probabilities = self._apply_output_activation(strong_logits)

        clip_pool = self.clip_pool(frame_features)
        clip_embedding = clip_pool["pooled"]
        weak_logits = self.weak_head(clip_embedding)  # [B, C]
        weak_probabilities = self._apply_output_activation(weak_logits)

        return {
            "input_spec": stem_out["input_spec"],
            "stem_feature": stem_out["stem_feature"],
            "frequency_attention": stem_out["frequency_attention"],
            "frame_features_pre_transformer": stem_out["frame_features_pre_transformer"],
            "frame_features": frame_features,
            "hidden_states": hidden_states if self.config.return_all_hidden_states else [frame_features],
            "strong_logits": strong_logits,
            "strong_probabilities": strong_probabilities,
            "weak_logits": weak_logits,
            "weak_probabilities": weak_probabilities,
            "clip_embedding": clip_embedding,
            "clip_attention": clip_pool["weights"],
        }

    def forward(self, input_spec: torch.Tensor) -> Dict[str, object]:
        return self.encode(input_spec)


def build_stage2_sed_guide(**kwargs) -> Stage2SEDGuideEncoder:
    return Stage2SEDGuideEncoder(**kwargs)


__all__ = [
    "Stage2SEDConfig",
    "Stage2SEDGuideEncoder",
    "build_stage2_sed_guide",
]
