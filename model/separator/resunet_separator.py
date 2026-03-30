"""ResUNet separator backbone for the ASD source-separation proxy.

This module defines the first separator backbone used in the new proxy-task
pipeline:

    mixture spectrogram -> ResUNet -> predicted target spectrogram

Design goals
------------
1. Be directly compatible with ``model/separator/frontend.py`` where the input
   is typically a single-channel spectrogram with shape ``[B, 1, F, T]``.
2. Reuse the low-level blocks from ``model/separator/resunet_blocks.py``.
3. Expose intermediate feature maps that can later be pooled/concatenated into
   ASD embeddings.
4. Support two practical output modes for the MVP:
   - ``mask``  : predict a ratio mask and multiply it with the input mixture
   - ``direct``: directly predict the target spectrogram

Recommended first usage
-----------------------
- Input representation: mixture magnitude spectrogram
- Output mode: ``mask``
- Output activation: ``sigmoid``
- ASD features: collect encoder outputs + bottleneck feature
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .resunet_blocks import (
    BlockConfig,
    BottleneckStage2d,
    ConvNormAct2d,
    DecoderStage2d,
    EncoderStage2d,
    SpectrogramHead2d,
    center_crop_or_pad_2d,
)


@dataclass(frozen=True)
class ResUNetSeparatorConfig:
    """Configuration for :class:`ResUNetSeparator`."""

    in_channels: int = 1
    encoder_channels: Tuple[int, ...] = (32, 64, 128)
    bottleneck_channels: Optional[int] = None
    stem_channels: Optional[int] = None
    block_config: BlockConfig = field(default_factory=BlockConfig)
    num_encoder_res_blocks: int = 2
    num_bottleneck_res_blocks: int = 2
    num_decoder_res_blocks: int = 2
    downsample_mode: str = "conv"
    upsample_mode: str = "bilinear"
    output_mode: str = "mask"
    output_activation: Optional[str] = None
    return_all_encoder_features: bool = True
    return_decoder_features: bool = False
    asd_feature_source: str = "encoder_bottleneck"


class ResUNetSeparator(nn.Module):
    """Lightweight ResUNet separator for the ASD source-separation pipeline.

    Input / output
    --------------
    Input:
        ``mix_spec`` with shape ``[B, 1, F, T]``.
    Output dictionary:
        - ``pred_spec``: predicted target spectrogram ``[B,1,F,T]``
        - ``pred_mask``: predicted mask ``[B,1,F,T]`` when ``output_mode='mask'``
        - ``encoder_features``: list of encoder feature maps
        - ``skip_features``: list of skip tensors used by the decoder
        - ``bottleneck_feature``: bottleneck tensor
        - ``decoder_features``: list of decoder feature maps
        - ``asd_feature_maps``: feature maps intended for later ASD embedding
    """

    VALID_OUTPUT_MODES = {"mask", "direct"}
    VALID_ASD_FEATURE_SOURCES = {"encoder_bottleneck", "all", "bottleneck_only"}

    def __init__(self, config: Optional[ResUNetSeparatorConfig] = None, **kwargs) -> None:
        super().__init__()

        if config is None:
            config = ResUNetSeparatorConfig(**kwargs)
        self.config = config

        if len(self.config.encoder_channels) < 1:
            raise ValueError("encoder_channels must contain at least one stage.")
        if self.config.output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported output_mode={self.config.output_mode!r}. "
                f"Supported: {sorted(self.VALID_OUTPUT_MODES)}"
            )
        if self.config.asd_feature_source not in self.VALID_ASD_FEATURE_SOURCES:
            raise ValueError(
                f"Unsupported asd_feature_source={self.config.asd_feature_source!r}. "
                f"Supported: {sorted(self.VALID_ASD_FEATURE_SOURCES)}"
            )

        cfg = self.config.block_config
        encoder_channels = tuple(int(c) for c in self.config.encoder_channels)
        stem_channels = encoder_channels[0] if self.config.stem_channels is None else int(self.config.stem_channels)
        bottleneck_channels = (
            encoder_channels[-1] if self.config.bottleneck_channels is None else int(self.config.bottleneck_channels)
        )
        self.encoder_channels = encoder_channels
        self.stem_channels = stem_channels
        self.bottleneck_channels = bottleneck_channels

        if self.config.output_activation is None:
            output_activation = "sigmoid" if self.config.output_mode == "mask" else "relu"
        else:
            output_activation = self.config.output_activation
        self.output_activation = output_activation

        self.stem = ConvNormAct2d(
            in_channels=self.config.in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=1,
            dilation=cfg.dilation,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_bias=cfg.use_bias,
        )

        encoder_stages: List[nn.Module] = []
        current_channels = stem_channels
        for i, out_channels in enumerate(encoder_channels):
            stage = EncoderStage2d(
                in_channels=current_channels,
                out_channels=out_channels,
                config=cfg,
                downsample=(i > 0),
                downsample_mode=self.config.downsample_mode,
                num_res_blocks=self.config.num_encoder_res_blocks,
            )
            encoder_stages.append(stage)
            current_channels = out_channels
        self.encoder = nn.ModuleList(encoder_stages)

        if current_channels != bottleneck_channels:
            self.pre_bottleneck = ConvNormAct2d(
                in_channels=current_channels,
                out_channels=bottleneck_channels,
                kernel_size=3,
                stride=1,
                dilation=cfg.dilation,
                norm_type=cfg.norm_type,
                activation=cfg.activation,
                dropout=cfg.dropout,
                use_bias=cfg.use_bias,
            )
        else:
            self.pre_bottleneck = nn.Identity()

        self.bottleneck = BottleneckStage2d(
            channels=bottleneck_channels,
            config=cfg,
            num_res_blocks=self.config.num_bottleneck_res_blocks,
        )

        decoder_stages: List[nn.Module] = []
        reversed_skips = list(reversed(encoder_channels[:-1]))
        decoder_out_channels = list(reversed(encoder_channels[:-1]))
        current_channels = bottleneck_channels
        for skip_channels, out_channels in zip(reversed_skips, decoder_out_channels):
            stage = DecoderStage2d(
                in_channels=current_channels,
                skip_channels=skip_channels,
                out_channels=out_channels,
                config=cfg,
                upsample_mode=self.config.upsample_mode,
                num_res_blocks=self.config.num_decoder_res_blocks,
            )
            decoder_stages.append(stage)
            current_channels = out_channels
        self.decoder = nn.ModuleList(decoder_stages)

        self.post_decoder = ConvNormAct2d(
            in_channels=current_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=1,
            dilation=cfg.dilation,
            norm_type=cfg.norm_type,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_bias=cfg.use_bias,
        )

        self.output_head = SpectrogramHead2d(
            in_channels=stem_channels,
            out_channels=1,
            output_activation=self.output_activation,
        )

    @staticmethod
    def _validate_input_spec(mix_spec: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(mix_spec):
            raise TypeError(f"Expected torch.Tensor, got {type(mix_spec)!r}.")
        if mix_spec.ndim == 3:
            mix_spec = mix_spec.unsqueeze(1)
        if mix_spec.ndim != 4:
            raise ValueError(
                f"Expected input spectrogram shape [B,1,F,T] or [B,F,T], got {tuple(mix_spec.shape)}."
            )
        if mix_spec.shape[1] != 1:
            raise ValueError(
                f"This MVP separator expects a single input channel; got shape {tuple(mix_spec.shape)}."
            )
        return mix_spec

    def _build_asd_feature_maps(
        self,
        encoder_features: Sequence[torch.Tensor],
        bottleneck_feature: torch.Tensor,
        decoder_features: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        source = self.config.asd_feature_source
        if source == "bottleneck_only":
            return [bottleneck_feature]
        if source == "encoder_bottleneck":
            feats = list(encoder_features) if self.config.return_all_encoder_features else [encoder_features[-1]]
            feats.append(bottleneck_feature)
            return feats
        if source == "all":
            feats = list(encoder_features) if self.config.return_all_encoder_features else [encoder_features[-1]]
            feats.append(bottleneck_feature)
            feats.extend(list(decoder_features))
            return feats
        raise RuntimeError(f"Unexpected asd_feature_source={source!r}.")

    def encode(self, mix_spec: torch.Tensor) -> Dict[str, object]:
        x = self._validate_input_spec(mix_spec)
        x = self.stem(x)

        encoder_features: List[torch.Tensor] = []
        skip_features: List[torch.Tensor] = []
        h = x
        for stage in self.encoder:
            h, skip = stage(h)
            encoder_features.append(h)
            skip_features.append(skip)

        h = self.pre_bottleneck(h)
        bottleneck_feature = self.bottleneck(h)

        return {
            "stem_feature": x,
            "encoder_features": encoder_features,
            "skip_features": skip_features,
            "bottleneck_feature": bottleneck_feature,
        }

    def decode(
        self,
        bottleneck_feature: torch.Tensor,
        skip_features: Sequence[torch.Tensor],
    ) -> Dict[str, object]:
        h = bottleneck_feature
        decoder_features: List[torch.Tensor] = []
        effective_skips = list(reversed(list(skip_features[:-1])))

        for stage, skip in zip(self.decoder, effective_skips):
            h = stage(h, skip)
            decoder_features.append(h)

        h = self.post_decoder(h)
        pred_head = self.output_head(h)
        return {
            "decoder_features": decoder_features,
            "pre_output_feature": h,
            "pred_head": pred_head,
        }

    def forward(self, mix_spec: torch.Tensor) -> Dict[str, object]:
        mix_spec = self._validate_input_spec(mix_spec)
        input_hw = mix_spec.shape[-2:]

        enc = self.encode(mix_spec)
        bottleneck_feature = enc["bottleneck_feature"]
        skip_features = enc["skip_features"]
        dec = self.decode(bottleneck_feature, skip_features)

        pred_head = center_crop_or_pad_2d(dec["pred_head"], target_hw=input_hw)

        if self.config.output_mode == "mask":
            pred_mask = pred_head
            pred_spec = pred_mask * mix_spec
        else:
            pred_mask = None
            pred_spec = pred_head

        decoder_features = dec["decoder_features"] if self.config.return_decoder_features else []
        asd_feature_maps = self._build_asd_feature_maps(
            encoder_features=enc["encoder_features"],
            bottleneck_feature=bottleneck_feature,
            decoder_features=decoder_features,
        )

        return {
            "input_spec": mix_spec,
            "pred_spec": pred_spec,
            "pred_mask": pred_mask,
            "stem_feature": enc["stem_feature"],
            "encoder_features": enc["encoder_features"],
            "skip_features": skip_features,
            "bottleneck_feature": bottleneck_feature,
            "decoder_features": decoder_features,
            "pre_output_feature": dec["pre_output_feature"],
            "asd_feature_maps": asd_feature_maps,
        }


def build_resunet_separator(**kwargs) -> ResUNetSeparator:
    return ResUNetSeparator(**kwargs)


__all__ = [
    "ResUNetSeparatorConfig",
    "ResUNetSeparator",
    "build_resunet_separator",
]
