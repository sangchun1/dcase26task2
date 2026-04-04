"""Extended ResUNet separator backbone for the ASD source-separation proxy.

This version keeps the original lightweight ResUNet path intact, but adds the
main hooks needed for the DCASE 2025 Task 4 2nd-place team's ideas:

1. bottleneck-side hidden-state injection
2. bottleneck / decoder Time-FiLM conditioning
3. optional DPRNN after the bottleneck
4. a helper for partial external pretrained loading

Important
---------
This file alone does **not** activate pretrained usage. It only provides the
model-side compatibility needed so that ``train_sep.py`` / ``ssmodule_sep.py``
can later load an external AudioSep-style checkpoint into the separator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .conditioning import (
    LatentFeatureInjection2d,
    LatentInjectionConfig,
    LearnedHiddenStateFusion,
    TimeFiLM2d,
    TimeFiLMConfig,
)
from .dprnn import DPRNN2d, DPRNNConfig
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

    # Morocutti-style extensions.
    use_dprnn: bool = False
    dprnn_hidden_size: int = 256
    dprnn_num_layers: int = 2
    dprnn_dropout: float = 0.0
    dprnn_bidirectional: bool = True
    dprnn_rnn_type: str = "gru"

    use_time_film: bool = False
    time_film_on_bottleneck: bool = True
    time_film_on_decoder: bool = False
    time_film_condition_dim: int = 1
    time_film_hidden_dim: int = 128
    time_film_num_layers: int = 2
    time_film_dropout: float = 0.0
    time_film_residual_gamma: bool = True

    use_latent_injection: bool = False
    latent_injection_input_dim: int = 256
    latent_injection_hidden_dim: int = 256
    latent_injection_num_hidden_states: int = 4


class ResUNetSeparator(nn.Module):
    """Lightweight ResUNet separator for the ASD source-separation pipeline.

    Input / output
    --------------
    Input:
        ``mix_spec`` with shape ``[B, 1, F, T]``.

    Optional conditioning:
        ``time_condition`` with shape ``[B, T]`` or ``[B, D, T]``
        ``hidden_states`` as a sequence of ``[B, T, D]`` tensors

    Output dictionary:
        - ``pred_spec``: predicted target spectrogram ``[B,1,F,T]``
        - ``pred_mask``: predicted mask ``[B,1,F,T]`` when ``output_mode='mask'``
        - ``encoder_features``: list of encoder feature maps
        - ``skip_features``: list of skip tensors used by the decoder
        - ``bottleneck_feature``: bottleneck tensor after optional conditioning
        - ``decoder_features``: list of decoder feature maps
        - ``asd_feature_maps``: feature maps intended for later ASD embedding
        - ``conditioning_debug``: optional info about FiLM / injection / fusion
    """

    VALID_OUTPUT_MODES = {"mask", "direct"}
    VALID_ASD_FEATURE_SOURCES = {"encoder_bottleneck", "all", "bottleneck_only"}

    def __init__(self, config: Optional[ResUNetSeparatorConfig] = None, **kwargs: Any) -> None:
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
        self.decoder_out_channels = tuple(decoder_out_channels)

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

        # Optional Morocutti-style extension modules.
        self.dprnn: Optional[nn.Module]
        if self.config.use_dprnn:
            self.dprnn = DPRNN2d(
                DPRNNConfig(
                    channels=bottleneck_channels,
                    hidden_size=self.config.dprnn_hidden_size,
                    num_layers=self.config.dprnn_num_layers,
                    dropout=self.config.dprnn_dropout,
                    bidirectional=self.config.dprnn_bidirectional,
                    rnn_type=self.config.dprnn_rnn_type,
                )
            )
        else:
            self.dprnn = None

        self.bottleneck_time_film: Optional[TimeFiLM2d]
        self.decoder_time_film: Optional[nn.ModuleList]
        if self.config.use_time_film:
            tf_cfg = TimeFiLMConfig(
                condition_dim=self.config.time_film_condition_dim,
                hidden_dim=self.config.time_film_hidden_dim,
                num_layers=self.config.time_film_num_layers,
                dropout=self.config.time_film_dropout,
                residual_gamma=self.config.time_film_residual_gamma,
            )
            self.bottleneck_time_film = TimeFiLM2d(bottleneck_channels, tf_cfg) if self.config.time_film_on_bottleneck else None
            self.decoder_time_film = (
                nn.ModuleList([TimeFiLM2d(channels, tf_cfg) for channels in self.decoder_out_channels])
                if self.config.time_film_on_decoder
                else None
            )
        else:
            self.bottleneck_time_film = None
            self.decoder_time_film = None

        self.hidden_state_fusion: Optional[LearnedHiddenStateFusion] = None
        self.latent_injection: Optional[LatentFeatureInjection2d]
        if self.config.use_latent_injection:
            self.hidden_state_fusion = LearnedHiddenStateFusion(self.config.latent_injection_num_hidden_states)
            self.latent_injection = LatentFeatureInjection2d(
                LatentInjectionConfig(
                    input_dim=self.config.latent_injection_input_dim,
                    target_channels=bottleneck_channels,
                    hidden_dim=self.config.latent_injection_hidden_dim,
                )
            )
        else:
            self.latent_injection = None

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

    def _maybe_fuse_hidden_states(
        self,
        hidden_states: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.config.use_latent_injection:
            return None, {}
        if hidden_states is None or len(hidden_states) == 0:
            return None, {}

        if self.hidden_state_fusion is None or self.hidden_state_fusion.num_states != len(hidden_states):
            # Lazily adapt to the actual guide depth if it differs from config.
            self.hidden_state_fusion = LearnedHiddenStateFusion(len(hidden_states)).to(hidden_states[0].device)

        fusion_out = self.hidden_state_fusion(hidden_states)
        return fusion_out["fused_hidden_state"], fusion_out

    def _apply_bottleneck_conditioning(
        self,
        bottleneck_feature: torch.Tensor,
        *,
        time_condition: Optional[torch.Tensor] = None,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        debug: Dict[str, Any] = {}
        h = bottleneck_feature

        fused_hidden_state, fusion_debug = self._maybe_fuse_hidden_states(hidden_states)
        if fused_hidden_state is not None and self.latent_injection is not None:
            inj_out = self.latent_injection(h, fused_hidden_state)
            h = inj_out["output"]
            debug["hidden_state_fusion"] = fusion_debug
            debug["latent_injection"] = inj_out

        if self.dprnn is not None:
            dprnn_out = self.dprnn(h)
            if isinstance(dprnn_out, dict):
                h = dprnn_out.get("output", h)
                debug["dprnn"] = dprnn_out
            else:
                h = dprnn_out

        if self.bottleneck_time_film is not None and time_condition is not None:
            film_out = self.bottleneck_time_film(h, time_condition)
            h = film_out["output"]
            debug["bottleneck_time_film"] = film_out

        return h, debug

    def _apply_decoder_conditioning(
        self,
        feature_map: torch.Tensor,
        stage_index: int,
        *,
        time_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.decoder_time_film is None or time_condition is None:
            return feature_map, {}
        film_out = self.decoder_time_film[stage_index](feature_map, time_condition)
        return film_out["output"], film_out

    def encode(
        self,
        mix_spec: torch.Tensor,
        *,
        time_condition: Optional[torch.Tensor] = None,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, object]:
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
        bottleneck_feature, conditioning_debug = self._apply_bottleneck_conditioning(
            bottleneck_feature,
            time_condition=time_condition,
            hidden_states=hidden_states,
        )
        return {
            "stem_feature": x,
            "encoder_features": encoder_features,
            "skip_features": skip_features,
            "bottleneck_feature": bottleneck_feature,
            "conditioning_debug": conditioning_debug,
        }

    def decode(
        self,
        bottleneck_feature: torch.Tensor,
        skip_features: Sequence[torch.Tensor],
        *,
        time_condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        h = bottleneck_feature
        decoder_features: List[torch.Tensor] = []
        decoder_conditioning_debug: List[Dict[str, torch.Tensor]] = []
        effective_skips = list(reversed(list(skip_features[:-1])))

        for stage_index, (stage, skip) in enumerate(zip(self.decoder, effective_skips)):
            h = stage(h, skip)
            h, stage_debug = self._apply_decoder_conditioning(
                h,
                stage_index,
                time_condition=time_condition,
            )
            decoder_features.append(h)
            decoder_conditioning_debug.append(stage_debug)

        h = self.post_decoder(h)
        pred_head = self.output_head(h)
        return {
            "decoder_features": decoder_features,
            "decoder_conditioning_debug": decoder_conditioning_debug,
            "pre_output_feature": h,
            "pred_head": pred_head,
        }

    def forward(
        self,
        mix_spec: torch.Tensor,
        *,
        time_condition: Optional[torch.Tensor] = None,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
        return_conditioning_debug: bool = True,
    ) -> Dict[str, object]:
        mix_spec = self._validate_input_spec(mix_spec)
        input_hw = mix_spec.shape[-2:]

        enc = self.encode(
            mix_spec,
            time_condition=time_condition,
            hidden_states=hidden_states,
        )
        bottleneck_feature = enc["bottleneck_feature"]
        skip_features = enc["skip_features"]

        dec = self.decode(
            bottleneck_feature,
            skip_features,
            time_condition=time_condition,
        )
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

        output: Dict[str, object] = {
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
        if return_conditioning_debug:
            output["conditioning_debug"] = {
                "encode": enc.get("conditioning_debug", {}),
                "decode": dec.get("decoder_conditioning_debug", []),
            }
        return output

    @staticmethod
    def _adapt_input_channel_weight(
        source: torch.Tensor,
        target_shape: torch.Size,
    ) -> Optional[torch.Tensor]:
        """Adapt an external conv-like weight to a smaller input-channel count.

        This is mainly for importing Morocutti/AudiosSep-style separator weights
        trained on 4-channel mixtures (or 5 channels for iterative refinement)
        into this mono ASD separator. When only the input-channel dimension is
        different, the source channels are reduced by averaging.

        Special case:
            source_in=5 and target_in=1 -> average only the first 4 channels,
            so iterative checkpoints do not mix the extra refinement estimate
            channel into the mono front-end initialization.
        """

        if source.ndim < 3 or len(target_shape) != source.ndim:
            return None
        if target_shape[1] >= source.shape[1]:
            return None
        if tuple(source.shape[0:1]) != tuple(target_shape[0:1]):
            return None
        if tuple(source.shape[2:]) != tuple(target_shape[2:]):
            return None

        target_in = int(target_shape[1])
        source_in = int(source.shape[1])
        if target_in <= 0 or source_in <= 0:
            return None

        reduce_source = source
        if source_in == 5 and target_in == 1:
            reduce_source = source[:, :4, ...]
            source_in = 4

        if target_in == source_in:
            return reduce_source

        if target_in == 1:
            return reduce_source.mean(dim=1, keepdim=True)

        if source_in % target_in != 0:
            return None

        group_size = source_in // target_in
        new_shape = (reduce_source.shape[0], target_in, group_size, *reduce_source.shape[2:])
        return reduce_source.reshape(new_shape).mean(dim=2)

    def load_pretrained_separator_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict_backbone: bool = False,
        strip_prefixes: Iterable[str] = (
            "separator.",
            "model.separator.",
            "module.separator.",
            "module.",
        ),
    ) -> Dict[str, Any]:
        """Partially load a pretrained separator checkpoint.

        This helper is intended for later AudioSep-style initialization. It
        filters a foreign state dict down to keys that exist in the current
        separator and either:

        1. match the expected tensor shape exactly, or
        2. can be adapted by reducing only the input-channel dimension.

        The second path is important for transferring Morocutti-style separator
        checkpoints trained with 4 input channels into this mono ASD backbone,
        while still keeping the no-pretrained baseline completely unchanged.

        Parameters
        ----------
        state_dict:
            Raw state dict from an external checkpoint, or the ``state_dict``
            field already extracted from one.
        strict_backbone:
            If ``True``, call ``load_state_dict(..., strict=True)`` on the
            filtered backbone keys. In practice ``False`` is recommended for the
            first AudioSep transfer attempt.
        strip_prefixes:
            Candidate key prefixes to remove before matching against the current
            separator keys.
        """

        if not isinstance(state_dict, Mapping):
            raise TypeError(f"state_dict must be a mapping, got {type(state_dict)!r}.")

        own_state = self.state_dict()
        normalized_state: Dict[str, torch.Tensor] = {}
        unexpected_source_keys: List[str] = []
        skipped_shape_keys: List[str] = []
        adapted_shape_keys: List[str] = []

        for raw_key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue

            candidate_keys = [raw_key]
            for prefix in strip_prefixes:
                if raw_key.startswith(prefix):
                    candidate_keys.append(raw_key[len(prefix) :])

            matched_key: Optional[str] = None
            for key in candidate_keys:
                if key in own_state:
                    matched_key = key
                    break

            if matched_key is None:
                unexpected_source_keys.append(raw_key)
                continue

            target_tensor = own_state[matched_key]
            if tuple(value.shape) == tuple(target_tensor.shape):
                normalized_state[matched_key] = value
                continue

            adapted_value = self._adapt_input_channel_weight(value, target_tensor.shape)
            if adapted_value is not None and tuple(adapted_value.shape) == tuple(target_tensor.shape):
                normalized_state[matched_key] = adapted_value.to(dtype=target_tensor.dtype)
                adapted_shape_keys.append(raw_key)
                continue

            skipped_shape_keys.append(raw_key)

        incompatible = self.load_state_dict(normalized_state, strict=strict_backbone)
        return {
            "num_loaded_tensors": len(normalized_state),
            "num_adapted_tensors": len(adapted_shape_keys),
            "loaded_keys": sorted(normalized_state.keys()),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "unexpected_source_keys": unexpected_source_keys,
            "skipped_shape_keys": skipped_shape_keys,
            "adapted_shape_keys": adapted_shape_keys,
        }


def build_resunet_separator(**kwargs: Any) -> ResUNetSeparator:
    return ResUNetSeparator(**kwargs)


__all__ = [
    "ResUNetSeparatorConfig",
    "ResUNetSeparator",
    "build_resunet_separator",
]
