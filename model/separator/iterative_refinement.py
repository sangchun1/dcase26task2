from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class IterativeRefinementConfig:
    """Configuration for iterative separator refinement.

    This wrapper is intentionally architecture-agnostic so it can sit on top of
    the current ASD proxy separator with minimal changes.

    The default usage pattern is:
        1) First iteration receives the original mixture plus an all-zero
           refinement side-channel.
        2) Later iterations receive the original mixture plus the previous
           iteration's prediction (typically ``pred_spec``).
        3) A lightweight input adapter maps the concatenated tensor back to the
           base separator input channel size.

    Parameters
    ----------
    enabled:
        Global on/off switch.
    num_iterations:
        Total number of refinement steps. ``1`` disables refinement while still
        allowing the wrapper to be used uniformly.
    detach_between_iterations:
        Whether to detach the previous prediction before feeding it into the
        next step. This matches the common training choice used to avoid very
        deep cross-iteration gradient chains.
    base_input_channels:
        Number of channels expected by the underlying separator input.
    refinement_channels:
        Number of side-channel refinement maps concatenated to the mixture.
        For the current monaural spectrogram pipeline this is usually ``1``.
    refinement_signal_key:
        Key to extract from a dict-like separator output. ``pred_spec`` is the
        recommended default for spectrogram-domain refinement.
    adapter_hidden_channels:
        Hidden channels used by the small input adapter.
    adapter_num_layers:
        Number of 1x1 convolutional layers used by the adapter. Must be >= 1.
    adapter_activation:
        Activation used in intermediate adapter layers. Supported:
        ``relu``, ``gelu``, ``silu``.
    residual_to_base_input:
        If ``True``, the adapter output is added to the original mixture input.
    return_history:
        If ``True``, forward returns per-iteration outputs in
        ``output['iteration_outputs']`` when the base separator returns a dict.
    """

    enabled: bool = True
    num_iterations: int = 2
    detach_between_iterations: bool = True
    base_input_channels: int = 1
    refinement_channels: int = 1
    refinement_signal_key: str = "pred_spec"
    adapter_hidden_channels: int = 16
    adapter_num_layers: int = 2
    adapter_activation: str = "silu"
    residual_to_base_input: bool = True
    return_history: bool = False

    def validate(self) -> None:
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.base_input_channels < 1:
            raise ValueError("base_input_channels must be >= 1")
        if self.refinement_channels < 1:
            raise ValueError("refinement_channels must be >= 1")
        if self.adapter_num_layers < 1:
            raise ValueError("adapter_num_layers must be >= 1")
        if self.adapter_activation not in {"relu", "gelu", "silu"}:
            raise ValueError(
                "adapter_activation must be one of {'relu', 'gelu', 'silu'}"
            )


class _Activation(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        if name == "relu":
            self.fn = nn.ReLU(inplace=True)
        elif name == "gelu":
            self.fn = nn.GELU()
        elif name == "silu":
            self.fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class IterativeRefinementInputAdapter(nn.Module):
    """Maps [mixture, previous_prediction] back to the separator input space."""

    def __init__(self, config: IterativeRefinementConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        in_channels = config.base_input_channels + config.refinement_channels
        out_channels = config.base_input_channels
        hidden_channels = max(config.adapter_hidden_channels, out_channels)

        layers: List[nn.Module] = []
        if config.adapter_num_layers == 1:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))
        else:
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True))
            layers.append(_Activation(config.adapter_activation))
            for _ in range(config.adapter_num_layers - 2):
                layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True))
                layers.append(_Activation(config.adapter_activation))
            layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, mixture_input: Tensor, refinement_signal: Tensor) -> Tensor:
        if mixture_input.ndim != 4:
            raise ValueError(
                f"mixture_input must be 4D [B, C, F, T], got shape {tuple(mixture_input.shape)}"
            )
        if refinement_signal.ndim != 4:
            raise ValueError(
                f"refinement_signal must be 4D [B, C, F, T], got shape {tuple(refinement_signal.shape)}"
            )

        refinement_signal = _resize_like(refinement_signal, mixture_input)
        refinement_signal = _match_channels(
            refinement_signal,
            target_channels=self.config.refinement_channels,
        )

        x = torch.cat([mixture_input, refinement_signal], dim=1)
        x = self.net(x)
        if self.config.residual_to_base_input:
            x = x + mixture_input
        return x


class IterativeRefinementWrapper(nn.Module):
    """Wrap a base separator with iterative refinement.

    Notes
    -----
    - The wrapped separator is expected to consume a 4D tensor with shape
      ``[B, C, F, T]``.
    - The wrapped separator may return either a tensor or a dict. If a dict is
      returned, the refinement tensor is extracted from
      ``config.refinement_signal_key``.
    - The final forward output preserves the base separator output type. When
      the base separator returns a dict, optional iteration history is appended.
    """

    def __init__(
        self,
        separator: nn.Module,
        config: Optional[IterativeRefinementConfig] = None,
    ) -> None:
        super().__init__()
        self.separator = separator
        self.config = config or IterativeRefinementConfig()
        self.config.validate()
        self.input_adapter = IterativeRefinementInputAdapter(self.config)

    def forward(self, mixture_input: Tensor, *args: Any, **kwargs: Any) -> Any:
        if mixture_input.ndim != 4:
            raise ValueError(
                f"mixture_input must be 4D [B, C, F, T], got shape {tuple(mixture_input.shape)}"
            )

        if not self.config.enabled or self.config.num_iterations == 1:
            return self.separator(mixture_input, *args, **kwargs)

        previous_prediction = self._make_initial_refinement_signal(mixture_input)
        history: List[Any] = []
        separator_output: Any = None

        for _ in range(self.config.num_iterations):
            adapted_input = self.input_adapter(mixture_input, previous_prediction)
            separator_output = self.separator(adapted_input, *args, **kwargs)
            if self.config.return_history:
                history.append(separator_output)

            previous_prediction = self._extract_refinement_signal(separator_output)
            previous_prediction = _resize_like(previous_prediction, mixture_input)
            previous_prediction = _match_channels(
                previous_prediction,
                target_channels=self.config.refinement_channels,
            )
            if self.config.detach_between_iterations:
                previous_prediction = previous_prediction.detach()

        if isinstance(separator_output, dict):
            final_output = dict(separator_output)
            final_output["refined_input"] = adapted_input
            if self.config.return_history:
                final_output["iteration_outputs"] = history
            return final_output

        return separator_output

    def _make_initial_refinement_signal(self, mixture_input: Tensor) -> Tensor:
        bsz, _, freq_bins, num_frames = mixture_input.shape
        return torch.zeros(
            bsz,
            self.config.refinement_channels,
            freq_bins,
            num_frames,
            device=mixture_input.device,
            dtype=mixture_input.dtype,
        )

    def _extract_refinement_signal(self, separator_output: Any) -> Tensor:
        if torch.is_tensor(separator_output):
            return separator_output

        if isinstance(separator_output, dict):
            key = self.config.refinement_signal_key
            if key not in separator_output:
                available = ", ".join(separator_output.keys())
                raise KeyError(
                    f"Separator output does not contain key '{key}'. Available keys: {available}"
                )
            value = separator_output[key]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Separator output['{key}'] must be a tensor, got {type(value).__name__}"
                )
            return value

        if hasattr(separator_output, self.config.refinement_signal_key):
            value = getattr(separator_output, self.config.refinement_signal_key)
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Separator attribute '{self.config.refinement_signal_key}' must be a tensor, "
                    f"got {type(value).__name__}"
                )
            return value

        raise TypeError(
            "Unsupported separator output type for iterative refinement: "
            f"{type(separator_output).__name__}"
        )


def build_iterative_refinement_wrapper(
    separator: nn.Module,
    config: Optional[IterativeRefinementConfig] = None,
) -> IterativeRefinementWrapper:
    return IterativeRefinementWrapper(separator=separator, config=config)


def _resize_like(x: Tensor, reference: Tensor) -> Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _match_channels(x: Tensor, target_channels: int) -> Tensor:
    if x.shape[1] == target_channels:
        return x

    if target_channels == 1:
        return x.mean(dim=1, keepdim=True)

    if x.shape[1] > target_channels:
        return x[:, :target_channels]

    repeats = target_channels // x.shape[1]
    remainder = target_channels % x.shape[1]
    out = x.repeat(1, repeats, 1, 1)
    if remainder > 0:
        out = torch.cat([out, x[:, :remainder]], dim=1)
    return out


__all__ = [
    "IterativeRefinementConfig",
    "IterativeRefinementInputAdapter",
    "IterativeRefinementWrapper",
    "build_iterative_refinement_wrapper",
]
