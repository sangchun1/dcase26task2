"""Building blocks for a lightweight ResUNet separator backbone.

This module contains reusable 2D blocks for the first source-separation MVP.
It is designed to work with the STFT frontend in ``model/separator/frontend.py``
where separator inputs have shape ``[B, 1, F, T]``.

Design goals
------------
1. Keep the backbone simple and stable for a first ASD separation proxy.
2. Make encoder / decoder construction in ``resunet_separator.py`` easy.
3. Preserve enough flexibility to later add DPRNN, FiLM, or iterative
   refinement without rewriting the low-level blocks.

Recommended first use
---------------------
- Input: magnitude spectrogram ``[B, 1, F, T]``
- Backbone: ResUNet made from ``EncoderStage2d`` and ``DecoderStage2d``
- ASD features: collect outputs from encoder stages and bottleneck
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _make_norm_2d(num_channels: int, norm_type: str = "bn") -> nn.Module:
    """Create a 2D normalization layer.

    Parameters
    ----------
    num_channels:
        Number of channels in the feature map.
    norm_type:
        One of ``"bn"``, ``"in"``, ``"gn"``, or ``"none"``.
    """
    norm_type = norm_type.lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "gn":
        # A conservative default that works for many channel sizes.
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type={norm_type!r}.")


def _make_activation(name: str = "prelu", num_parameters: int = 1) -> nn.Module:
    """Create an activation layer."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "prelu":
        return nn.PReLU(num_parameters=num_parameters)
    raise ValueError(f"Unsupported activation={name!r}.")


def center_crop_or_pad_2d(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Center-crop or zero-pad a feature map to the requested spatial size.

    Parameters
    ----------
    x:
        Tensor with shape ``[B, C, H, W]``.
    target_hw:
        Target spatial size ``(H_target, W_target)``.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    _, _, h, w = x.shape

    # Crop height.
    if h > target_h:
        start = (h - target_h) // 2
        x = x[:, :, start : start + target_h, :]
    # Crop width.
    if w > target_w:
        start = (w - target_w) // 2
        x = x[:, :, :, start : start + target_w]

    _, _, h, w = x.shape

    # Pad height.
    if h < target_h:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
    else:
        pad_top = pad_bottom = 0

    # Pad width.
    if w < target_w:
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
    else:
        pad_left = pad_right = 0

    if pad_top or pad_bottom or pad_left or pad_right:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    return x


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockConfig:
    """Configuration shared by the ResUNet blocks.

    Attributes
    ----------
    kernel_size:
        Convolution kernel size used by most blocks.
    dilation:
        Dilation used by residual convolutions.
    norm_type:
        ``"bn"``, ``"in"``, ``"gn"``, or ``"none"``.
    activation:
        ``"prelu"``, ``"relu"``, ``"gelu"``, etc.
    dropout:
        Spatial dropout probability. Set 0.0 to disable.
    use_bias:
        Whether Conv2d / ConvTranspose2d uses bias.
    """

    kernel_size: int = 3
    dilation: int = 1
    norm_type: str = "bn"
    activation: str = "prelu"
    dropout: float = 0.0
    use_bias: bool = False


# -----------------------------------------------------------------------------
# Primitive blocks
# -----------------------------------------------------------------------------

class ConvNormAct2d(nn.Module):
    """Conv2d -> Norm -> Activation block.

    Input/output shape
    ------------------
    ``[B, C_in, H, W] -> [B, C_out, H_out, W_out]``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_type: str = "bn",
        activation: str = "prelu",
        dropout: float = 0.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = _make_norm_2d(out_channels, norm_type=norm_type)
        self.act = _make_activation(activation, num_parameters=out_channels if activation == "prelu" else 1)
        self.drop = nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualConvBlock2d(nn.Module):
    """Two-layer residual convolution block.

    This is the main workhorse block for the separator encoder/decoder.

    Structure
    ---------
    input
      -> ConvNormAct
      -> ConvNorm
      -> residual add (with optional 1x1 projection)
      -> activation

    It preserves spatial size when ``stride=1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: Optional[BlockConfig] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        cfg = BlockConfig() if config is None else config
        padding = ((cfg.kernel_size - 1) // 2) * cfg.dilation

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=cfg.kernel_size,
            stride=stride,
            padding=padding,
            dilation=cfg.dilation,
            bias=cfg.use_bias,
        )
        self.norm1 = _make_norm_2d(out_channels, norm_type=cfg.norm_type)
        self.act1 = _make_activation(cfg.activation, num_parameters=out_channels if cfg.activation == "prelu" else 1)
        self.drop = nn.Dropout2d(p=cfg.dropout) if cfg.dropout and cfg.dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=cfg.kernel_size,
            stride=1,
            padding=padding,
            dilation=cfg.dilation,
            bias=cfg.use_bias,
        )
        self.norm2 = _make_norm_2d(out_channels, norm_type=cfg.norm_type)
        self.out_act = _make_activation(cfg.activation, num_parameters=out_channels if cfg.activation == "prelu" else 1)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=cfg.use_bias),
                _make_norm_2d(out_channels, norm_type=cfg.norm_type),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.out_act(out)
        return out


class Downsample2d(nn.Module):
    """Downsample a feature map by a factor of 2.

    Two modes are provided:
    - ``"conv"``: learnable stride-2 ConvNormAct block
    - ``"avgpool"``: average pool followed by a 1x1 projection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "conv",
        norm_type: str = "bn",
        activation: str = "prelu",
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode == "conv":
            self.net = ConvNormAct2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                activation=activation,
                use_bias=use_bias,
            )
        elif mode == "avgpool":
            self.net = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias),
                _make_norm_2d(out_channels, norm_type=norm_type),
                _make_activation(activation, num_parameters=out_channels if activation == "prelu" else 1),
            )
        else:
            raise ValueError("Downsample mode must be 'conv' or 'avgpool'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Upsample2d(nn.Module):
    """Upsample a feature map by a factor of 2.

    Two modes are provided:
    - ``"transpose"``: ConvTranspose2d
    - ``"bilinear"``: interpolation + 1x1 projection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "transpose",
        norm_type: str = "bn",
        activation: str = "prelu",
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode == "transpose":
            self.net = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=use_bias,
                ),
                _make_norm_2d(out_channels, norm_type=norm_type),
                _make_activation(activation, num_parameters=out_channels if activation == "prelu" else 1),
            )
        elif mode == "bilinear":
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias),
                _make_norm_2d(out_channels, norm_type=norm_type),
                _make_activation(activation, num_parameters=out_channels if activation == "prelu" else 1),
            )
        else:
            raise ValueError("Upsample mode must be 'transpose' or 'bilinear'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Higher-level encoder / decoder stages
# -----------------------------------------------------------------------------

class EncoderStage2d(nn.Module):
    """A single ResUNet encoder stage.

    Parameters
    ----------
    in_channels, out_channels:
        Channel sizes for the stage.
    config:
        Shared block configuration.
    downsample:
        Whether to downsample at the start of the stage.
    downsample_mode:
        ``"conv"`` or ``"avgpool"``.

    Returns
    -------
    y:
        Output feature map of the stage.
    skip:
        Skip feature used by the decoder. Here we simply return the same tensor
        after the residual blocks because that is usually the most useful signal
        for U-Net skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: Optional[BlockConfig] = None,
        downsample: bool = False,
        downsample_mode: str = "conv",
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        cfg = BlockConfig() if config is None else config

        layers = []
        if downsample:
            layers.append(
                Downsample2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    mode=downsample_mode,
                    norm_type=cfg.norm_type,
                    activation=cfg.activation,
                    use_bias=cfg.use_bias,
                )
            )
            current_channels = out_channels
        else:
            current_channels = in_channels
            if in_channels != out_channels:
                layers.append(
                    ConvNormAct2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        dilation=cfg.dilation,
                        norm_type=cfg.norm_type,
                        activation=cfg.activation,
                        dropout=cfg.dropout,
                        use_bias=cfg.use_bias,
                    )
                )
                current_channels = out_channels

        for _ in range(num_res_blocks):
            layers.append(ResidualConvBlock2d(current_channels, out_channels, config=cfg))
            current_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.net(x)
        skip = y
        return y, skip


class BottleneckStage2d(nn.Module):
    """Bottleneck block placed between encoder and decoder."""

    def __init__(
        self,
        channels: int,
        config: Optional[BlockConfig] = None,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        cfg = BlockConfig() if config is None else config
        self.blocks = nn.Sequential(
            *[ResidualConvBlock2d(channels, channels, config=cfg) for _ in range(num_res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DecoderStage2d(nn.Module):
    """A single ResUNet decoder stage.

    Workflow
    --------
    1. Upsample decoder feature map.
    2. Center-crop/pad to match the skip spatial size.
    3. Merge with skip connection using concatenation.
    4. Fuse with one projection block and residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        config: Optional[BlockConfig] = None,
        upsample_mode: str = "transpose",
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        cfg = BlockConfig() if config is None else config

        self.upsample = Upsample2d(
            in_channels=in_channels,
            out_channels=out_channels,
            mode=upsample_mode,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
            use_bias=cfg.use_bias,
        )

        fuse_in_channels = out_channels + skip_channels
        self.fuse = ConvNormAct2d(
            in_channels=fuse_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=cfg.dilation,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_bias=cfg.use_bias,
        )

        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock2d(out_channels, out_channels, config=cfg) for _ in range(num_res_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = center_crop_or_pad_2d(x, target_hw=(skip.shape[-2], skip.shape[-1]))
        merged = torch.cat([x, skip], dim=1)
        y = self.fuse(merged)
        y = self.res_blocks(y)
        return y


# -----------------------------------------------------------------------------
# Output heads
# -----------------------------------------------------------------------------

class SpectrogramHead2d(nn.Module):
    """Project decoder features to a single-channel separator output.

    Parameters
    ----------
    in_channels:
        Number of channels from the last decoder stage.
    out_channels:
        Usually ``1`` for the current MVP.
    output_activation:
        - ``None`` / ``"identity"``: raw prediction
        - ``"sigmoid"``: useful for ratio-mask style output
        - ``"relu"`` / ``"softplus"``: non-negative magnitude output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if output_activation is None or output_activation == "identity":
            self.out_act = nn.Identity()
        elif output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif output_activation == "softplus":
            self.out_act = nn.Softplus()
        else:
            raise ValueError(f"Unsupported output_activation={output_activation!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.proj(x))


__all__ = [
    "BlockConfig",
    "ConvNormAct2d",
    "ResidualConvBlock2d",
    "Downsample2d",
    "Upsample2d",
    "EncoderStage2d",
    "BottleneckStage2d",
    "DecoderStage2d",
    "SpectrogramHead2d",
    "center_crop_or_pad_2d",
]
