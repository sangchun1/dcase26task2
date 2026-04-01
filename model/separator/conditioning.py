"""Conditioning modules for Morocutti-style temporal guidance in separation.

This file is designed to be the first building block for adding the DCASE 2025
2nd-place team's conditioning ideas into the ASD source-separation proxy.

Included modules
----------------
1. ``TimeFiLM2d``
   Time-varying FiLM modulation for 2D separator feature maps.
2. ``LearnedHiddenStateFusion``
   Softmax-weighted fusion of multiple hidden states (for example, Transformer
   block outputs from a stage-2 SED / guide encoder).
3. ``LatentFeatureInjection2d``
   Projection of fused sequence features into a separator latent tensor followed
   by residual element-wise addition.
4. ``gather_class_probability_map``
   Utility for selecting a target class probability trajectory from a
   frame-wise class probability tensor.

The module is intentionally self-contained so it can be imported later by
``resunet_separator.py`` and ``ssmodule_sep.py`` without forcing a particular
SED implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLikeIndex = Union[int, torch.Tensor]


@dataclass(frozen=True)
class TimeFiLMConfig:
    """Configuration for :class:`TimeFiLM2d`.

    Parameters
    ----------
    condition_dim:
        Number of channels in the temporal conditioning input. A frame-wise
        probability map usually uses ``1``.
    hidden_dim:
        Internal hidden size of the temporal projector.
    num_layers:
        Number of Conv1d projector blocks before the final gamma/beta head.
    dropout:
        Dropout probability inside the projector.
    residual_gamma:
        If ``True``, gamma is parameterized around identity as ``1 + tanh(.)``.
    use_bias:
        Whether Conv1d / Linear projections use bias terms.
    """

    condition_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    residual_gamma: bool = True
    use_bias: bool = True


@dataclass(frozen=True)
class LatentInjectionConfig:
    """Configuration for :class:`LatentFeatureInjection2d`.

    Parameters
    ----------
    input_dim:
        Hidden dimension of the fused sequence input ``[B, T, D]``.
    target_channels:
        Channel dimension of the separator feature map to inject into.
    hidden_dim:
        Hidden dimension used before projecting to ``target_channels``.
    add_residual:
        If ``True``, output is ``feature + gate * projected_condition``.
    post_conv:
        If ``True``, run a 1x1 Conv2d after broadcasting the temporal sequence
        over frequency.
    """

    input_dim: int
    target_channels: int
    hidden_dim: int = 256
    add_residual: bool = True
    post_conv: bool = True
    use_bias: bool = True


def _ensure_2d_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(feature_map):
        raise TypeError(f"Expected torch.Tensor, got {type(feature_map)!r}.")
    if feature_map.ndim != 4:
        raise ValueError(
            "Expected a 2D feature map with shape [B, C, F, T], "
            f"got {tuple(feature_map.shape)}."
        )
    return feature_map


def _ensure_temporal_condition(condition: torch.Tensor, *, condition_dim: Optional[int] = None) -> torch.Tensor:
    """Normalize temporal conditioning to ``[B, D, T]``.

    Supported inputs:
    - ``[B, T]``
    - ``[B, 1, T]`` or ``[B, D, T]``

    The function intentionally does not auto-interpret ``[B, T, D]`` because
    that can be ambiguous. Sequence-style hidden states should be explicitly
    converted before passing into FiLM.
    """

    if not torch.is_tensor(condition):
        raise TypeError(f"Expected torch.Tensor, got {type(condition)!r}.")

    if condition.ndim == 2:
        condition = condition.unsqueeze(1)  # [B, 1, T]
    elif condition.ndim != 3:
        raise ValueError(
            "Expected temporal condition shape [B, T] or [B, D, T], "
            f"got {tuple(condition.shape)}."
        )

    if condition_dim is not None and condition.shape[1] != condition_dim:
        raise ValueError(
            f"Expected condition_dim={condition_dim}, got shape {tuple(condition.shape)}."
        )

    return condition


def gather_class_probability_map(
    class_probabilities: torch.Tensor,
    class_index: TensorLikeIndex,
) -> torch.Tensor:
    """Select a target class probability trajectory.

    Parameters
    ----------
    class_probabilities:
        Tensor with shape ``[B, C, T]``.
    class_index:
        Either
        - an ``int`` for the same target class across the whole batch, or
        - a tensor of shape ``[B]`` with one target class index per sample.

    Returns
    -------
    torch.Tensor
        Selected probability map with shape ``[B, 1, T]``.
    """

    if class_probabilities.ndim != 3:
        raise ValueError(
            "Expected class_probabilities with shape [B, C, T], "
            f"got {tuple(class_probabilities.shape)}."
        )

    batch_size, num_classes, _ = class_probabilities.shape
    device = class_probabilities.device

    if isinstance(class_index, int):
        if class_index < 0 or class_index >= num_classes:
            raise IndexError(
                f"class_index={class_index} is out of range for {num_classes} classes."
            )
        selected = class_probabilities[:, class_index : class_index + 1, :]
        return selected

    if not torch.is_tensor(class_index):
        raise TypeError(
            "class_index must be an int or a torch.Tensor of shape [B]."
        )

    if class_index.ndim != 1 or class_index.shape[0] != batch_size:
        raise ValueError(
            "Batch-wise class_index tensor must have shape [B], "
            f"got {tuple(class_index.shape)}."
        )

    if class_index.dtype not in (torch.int32, torch.int64, torch.long):
        class_index = class_index.long()

    if (class_index < 0).any() or (class_index >= num_classes).any():
        raise IndexError(
            f"class_index tensor contains values outside [0, {num_classes - 1}]."
        )

    gather_index = class_index.to(device=device).view(batch_size, 1, 1).expand(-1, 1, class_probabilities.shape[-1])
    return torch.gather(class_probabilities, dim=1, index=gather_index)


class _TemporalProjector(nn.Module):
    """Small Conv1d projector used by Time-FiLM.

    Input / output shape: ``[B, D, T]``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        layers = []
        current = in_channels
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Conv1d(current, hidden_channels, kernel_size=3, padding=1, bias=use_bias),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_channels

        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Conv1d(current, out_channels, kernel_size=1, bias=use_bias)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)


class TimeFiLM2d(nn.Module):
    """Time-varying FiLM modulation for 2D separator feature maps.

    This module implements the main idea behind temporal conditioning:
    a frame-wise conditioning sequence controls the separator features via
    per-time, per-channel gamma / beta parameters.

    Expected shapes
    ---------------
    - feature_map: ``[B, C, F, T]``
    - condition : ``[B, T]`` or ``[B, D, T]``
    """

    def __init__(self, feature_channels: int, config: Optional[TimeFiLMConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TimeFiLMConfig(**kwargs)
        self.config = config
        self.feature_channels = int(feature_channels)

        self.projector = _TemporalProjector(
            in_channels=self.config.condition_dim,
            hidden_channels=self.config.hidden_dim,
            out_channels=2 * self.feature_channels,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_bias=self.config.use_bias,
        )

    def forward(self, feature_map: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_map = _ensure_2d_feature_map(feature_map)
        condition = _ensure_temporal_condition(condition, condition_dim=self.config.condition_dim)

        target_t = feature_map.shape[-1]
        if condition.shape[-1] != target_t:
            condition = F.interpolate(condition, size=target_t, mode="linear", align_corners=False)

        gamma_beta = self.projector(condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        gamma = torch.tanh(gamma)
        if self.config.residual_gamma:
            gamma = 1.0 + gamma
        beta = torch.tanh(beta)

        gamma_2d = gamma.unsqueeze(2)  # [B, C, 1, T]
        beta_2d = beta.unsqueeze(2)    # [B, C, 1, T]
        modulated = feature_map * gamma_2d + beta_2d

        return {
            "output": modulated,
            "gamma": gamma,
            "beta": beta,
            "resized_condition": condition,
        }


class LearnedHiddenStateFusion(nn.Module):
    """Softmax-weighted fusion of multiple hidden states.

    Intended use:
    - inputs are a list/tuple of hidden states from multiple Transformer blocks
    - each state has shape ``[B, T, D]``
    - the module learns one scalar importance per state and returns a weighted
      sum of the states
    """

    def __init__(self, num_states: int) -> None:
        super().__init__()
        if num_states < 1:
            raise ValueError("num_states must be >= 1.")
        self.num_states = int(num_states)
        self.layer_logits = nn.Parameter(torch.zeros(self.num_states))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(hidden_states) != self.num_states:
            raise ValueError(
                f"Expected {self.num_states} hidden states, got {len(hidden_states)}."
            )

        reference_shape: Optional[Tuple[int, ...]] = None
        for i, state in enumerate(hidden_states):
            if not torch.is_tensor(state):
                raise TypeError(f"Hidden state at index {i} is not a tensor.")
            if state.ndim != 3:
                raise ValueError(
                    "Each hidden state must have shape [B, T, D], "
                    f"but state {i} has shape {tuple(state.shape)}."
                )
            if reference_shape is None:
                reference_shape = tuple(state.shape)
            elif tuple(state.shape) != reference_shape:
                raise ValueError(
                    "All hidden states must have the same shape. "
                    f"Expected {reference_shape}, got {tuple(state.shape)} at index {i}."
                )

        weights = torch.softmax(self.layer_logits, dim=0)
        fused = torch.zeros_like(hidden_states[0])
        for weight, state in zip(weights, hidden_states):
            fused = fused + weight * state

        return {
            "fused_hidden_state": fused,
            "fusion_weights": weights,
        }


class LatentFeatureInjection2d(nn.Module):
    """Project a temporal hidden state into a 2D latent map and inject it.

    Expected shapes
    ---------------
    - feature_map        : ``[B, C, F, T]``
    - fused_hidden_state : ``[B, T, D]``

    The temporal hidden state is projected to ``target_channels``, resized to
    the separator's time resolution, broadcast across frequency, optionally
    refined with a 1x1 Conv2d, and then added to the separator feature map.
    A learnable scalar gate is zero-initialized so the module starts near an
    identity mapping.
    """

    def __init__(self, config: LatentInjectionConfig) -> None:
        super().__init__()
        self.config = config

        self.sequence_projector = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim, bias=self.config.use_bias),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.target_channels, bias=self.config.use_bias),
        )
        self.post_conv = (
            nn.Conv2d(self.config.target_channels, self.config.target_channels, kernel_size=1, bias=self.config.use_bias)
            if self.config.post_conv
            else nn.Identity()
        )
        self.gate = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        final_linear = self.sequence_projector[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)
        if isinstance(self.post_conv, nn.Conv2d):
            nn.init.zeros_(self.post_conv.weight)
            if self.post_conv.bias is not None:
                nn.init.zeros_(self.post_conv.bias)

    def forward(self, feature_map: torch.Tensor, fused_hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_map = _ensure_2d_feature_map(feature_map)
        if fused_hidden_state.ndim != 3:
            raise ValueError(
                "Expected fused_hidden_state with shape [B, T, D], "
                f"got {tuple(fused_hidden_state.shape)}."
            )

        batch_size, _, freq_bins, target_t = feature_map.shape
        if fused_hidden_state.shape[0] != batch_size:
            raise ValueError(
                "Batch dimension mismatch between feature_map and fused_hidden_state: "
                f"{feature_map.shape[0]} vs {fused_hidden_state.shape[0]}."
            )

        projected = self.sequence_projector(fused_hidden_state)   # [B, T, C]
        projected = projected.transpose(1, 2).contiguous()        # [B, C, T]

        if projected.shape[-1] != target_t:
            projected = F.interpolate(projected, size=target_t, mode="linear", align_corners=False)

        injection_map = projected.unsqueeze(2).expand(-1, -1, freq_bins, -1).contiguous()
        injection_map = self.post_conv(injection_map)

        gate = torch.tanh(self.gate)
        if self.config.add_residual:
            output = feature_map + gate * injection_map
        else:
            output = gate * injection_map

        return {
            "output": output,
            "injection_map": injection_map,
            "gate": gate,
        }


__all__ = [
    "TimeFiLMConfig",
    "LatentInjectionConfig",
    "TimeFiLM2d",
    "LearnedHiddenStateFusion",
    "LatentFeatureInjection2d",
    "gather_class_probability_map",
]
