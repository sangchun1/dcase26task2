"""Dual-Path RNN modules for Morocutti-style separator enhancement.

This module is designed to sit after a 2D separator bottleneck feature map with
shape ``[B, C, F, T]``. It applies recurrent modeling along

1. the time axis (for each frequency bin), and
2. the frequency axis (for each time frame),

then returns a feature map with the same shape.

The implementation is intentionally self-contained so it can be plugged into the
current ResUNet bottleneck without forcing other architectural changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DPRNNConfig:
    """Configuration for :class:`DPRNN2d`.

    Parameters
    ----------
    channels:
        Channel dimension ``C`` of the incoming feature map ``[B, C, F, T]``.
    hidden_size:
        Hidden size of each recurrent direction.
    num_layers:
        Number of stacked dual-path blocks.
    dropout:
        Dropout probability used in recurrent and feed-forward projections.
    bidirectional:
        Whether to use bidirectional recurrent layers.
    rnn_type:
        One of ``{"gru", "lstm"}``.
    use_bias:
        Whether recurrent and linear layers use bias terms.
    norm_type:
        One of ``{"layernorm", "groupnorm"}``.
    groupnorm_groups:
        Number of groups when ``norm_type="groupnorm"``.
    use_residual:
        Whether each dual-path block adds a residual connection.
    ff_multiplier:
        Hidden expansion ratio of the feed-forward refinement head.
    return_intermediate:
        Whether :class:`DPRNN2d` returns intermediate outputs from each block.
    """

    channels: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    bidirectional: bool = True
    rnn_type: str = "gru"
    use_bias: bool = True
    norm_type: str = "layernorm"
    groupnorm_groups: int = 8
    use_residual: bool = True
    ff_multiplier: float = 2.0
    return_intermediate: bool = False


class _SequenceNorm(nn.Module):
    """Normalize a sequence tensor with shape ``[N, L, C]``.

    - ``layernorm`` normalizes across the channel dimension per time step.
    - ``groupnorm`` is applied on the transposed ``[N, C, L]`` representation.
    """

    def __init__(self, channels: int, *, norm_type: str = "layernorm", groupnorm_groups: int = 8) -> None:
        super().__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(channels)
        elif norm_type == "groupnorm":
            groups = min(groupnorm_groups, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, channels)
        else:
            raise ValueError(f"Unsupported norm_type={norm_type!r}. Use 'layernorm' or 'groupnorm'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [N, L, C], got {tuple(x.shape)}.")
        if self.norm_type == "layernorm":
            return self.norm(x)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _FeedForwardRefine(nn.Module):
    """Small position-wise refinement head for sequence features."""

    def __init__(self, channels: int, *, multiplier: float = 2.0, dropout: float = 0.0, use_bias: bool = True) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * multiplier)))
        self.net = nn.Sequential(
            nn.Linear(channels, hidden, bias=use_bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels, bias=use_bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [N, L, C], got {tuple(x.shape)}.")
        return self.net(x)


class _PathRNN(nn.Module):
    """One recurrent path over a sequence representation ``[N, L, C]``."""

    VALID_RNN_TYPES = {"gru", "lstm"}

    def __init__(
        self,
        channels: int,
        *,
        hidden_size: int = 256,
        dropout: float = 0.0,
        bidirectional: bool = True,
        rnn_type: str = "gru",
        use_bias: bool = True,
        norm_type: str = "layernorm",
        groupnorm_groups: int = 8,
        ff_multiplier: float = 2.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in self.VALID_RNN_TYPES:
            raise ValueError(f"Unsupported rnn_type={rnn_type!r}. Use one of {sorted(self.VALID_RNN_TYPES)}.")
        self.channels = channels
        self.use_residual = use_residual
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=bidirectional,
            bias=use_bias,
        )
        out_channels = hidden_size * (2 if bidirectional else 1)
        self.project = nn.Sequential(
            nn.Linear(out_channels, channels, bias=use_bias),
            nn.Dropout(dropout),
        )
        self.norm = _SequenceNorm(channels, norm_type=norm_type, groupnorm_groups=groupnorm_groups)
        self.ff_norm = _SequenceNorm(channels, norm_type=norm_type, groupnorm_groups=groupnorm_groups)
        self.ff = _FeedForwardRefine(
            channels,
            multiplier=ff_multiplier,
            dropout=dropout,
            use_bias=use_bias,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [N, L, C], got {tuple(x.shape)}.")
        residual = x
        rnn_out, _ = self.rnn(x)
        y = self.project(rnn_out)
        if self.use_residual:
            y = y + residual
        y = self.norm(y)

        ff_residual = y
        y = self.ff(y)
        if self.use_residual:
            y = y + ff_residual
        y = self.ff_norm(y)
        return {"output": y, "residual": residual}


class DPRNNBlock2d(nn.Module):
    """One dual-path block on a feature map of shape ``[B, C, F, T]``.

    The block first applies recurrent modeling along the time axis for each
    frequency bin, then along the frequency axis for each time frame.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_size: int = 256,
        dropout: float = 0.0,
        bidirectional: bool = True,
        rnn_type: str = "gru",
        use_bias: bool = True,
        norm_type: str = "layernorm",
        groupnorm_groups: int = 8,
        ff_multiplier: float = 2.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.time_path = _PathRNN(
            channels,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_bias=use_bias,
            norm_type=norm_type,
            groupnorm_groups=groupnorm_groups,
            ff_multiplier=ff_multiplier,
            use_residual=use_residual,
        )
        self.freq_path = _PathRNN(
            channels,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_bias=use_bias,
            norm_type=norm_type,
            groupnorm_groups=groupnorm_groups,
            ff_multiplier=ff_multiplier,
            use_residual=use_residual,
        )

    @staticmethod
    def _check_input(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, F, T], got {tuple(x.shape)}.")
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._check_input(x)
        b, c, f, t = x.shape
        if c != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got input with channels={c}.")

        # Time path: each frequency bin is treated as one sequence over time.
        time_in = x.permute(0, 2, 3, 1).contiguous().view(b * f, t, c)  # [B*F, T, C]
        time_out = self.time_path(time_in)["output"]
        time_out_2d = time_out.view(b, f, t, c).permute(0, 3, 1, 2).contiguous()  # [B, C, F, T]

        # Frequency path: each time frame is treated as one sequence over frequency.
        freq_in = time_out_2d.permute(0, 3, 2, 1).contiguous().view(b * t, f, c)  # [B*T, F, C]
        freq_out = self.freq_path(freq_in)["output"]
        output = freq_out.view(b, t, f, c).permute(0, 3, 2, 1).contiguous()  # [B, C, F, T]

        return {
            "output": output,
            "time_out": time_out_2d,
        }


class DPRNN2d(nn.Module):
    """Stack of dual-path blocks for 2D separator bottleneck features."""

    def __init__(self, config: Optional[DPRNNConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = DPRNNConfig(**kwargs)
        self.config = config

        if self.config.channels < 1:
            raise ValueError("channels must be >= 1.")
        if self.config.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1.")
        if self.config.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if not (0.0 <= self.config.dropout < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.blocks = nn.ModuleList(
            [
                DPRNNBlock2d(
                    channels=self.config.channels,
                    hidden_size=self.config.hidden_size,
                    dropout=self.config.dropout,
                    bidirectional=self.config.bidirectional,
                    rnn_type=self.config.rnn_type,
                    use_bias=self.config.use_bias,
                    norm_type=self.config.norm_type,
                    groupnorm_groups=self.config.groupnorm_groups,
                    ff_multiplier=self.config.ff_multiplier,
                    use_residual=self.config.use_residual,
                )
                for _ in range(self.config.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, F, T], got {tuple(x.shape)}.")

        intermediates: List[torch.Tensor] = []
        out = x
        for block in self.blocks:
            out = block(out)["output"]
            if self.config.return_intermediate:
                intermediates.append(out)

        result: Dict[str, torch.Tensor] = {"output": out}
        if self.config.return_intermediate:
            result["intermediate_outputs"] = intermediates
        return result


def build_dprnn_2d(config: Optional[DPRNNConfig] = None, **kwargs) -> DPRNN2d:
    """Convenience builder for project-level imports."""

    return DPRNN2d(config=config, **kwargs)
