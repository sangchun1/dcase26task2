"""Feature pooling head for ASD embeddings from separator feature maps.

This module converts the intermediate feature maps produced by
``model/separator/resunet_separator.py`` into a fixed-dimensional embedding that
can be used by downstream anomaly scorers such as Mahalanobis distance.

Typical flow
------------
separator_out = separator(mix_spec)
head_out = feature_head(separator_out)
embedding = head_out["embedding"]

Design goals
------------
1. Accept the natural output format of ``ResUNetSeparator``.
2. Pool multi-scale 2D feature maps ``[B, C, F, T]`` into vectors.
3. Support simple MVP pooling first, while keeping an easy path toward richer
   embeddings later.
4. Return both the final embedding and per-level pooled vectors for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]
SeparatorOutput = Dict[str, object]


@dataclass(frozen=True)
class FeatureHeadConfig:
    """Configuration for :class:`SeparatorFeatureHead`.

    Parameters
    ----------
    pooling:
        How to pool each feature map over the frequency/time axes.
        Supported values:
        - ``"mean"``: global average pooling over ``F,T``
        - ``"max"``: global max pooling over ``F,T``
        - ``"mean_max"``: concatenate global mean and global max
        - ``"mean_std"``: concatenate global mean and standard deviation
        - ``"temporal_mean"``: mean over time after frequency aggregation
        - ``"temporal_max"``: max over time after frequency aggregation
        - ``"temporal_mean_max"``: concatenate temporal mean and temporal max
        - ``"temporal_mean_std"``: concatenate temporal mean and temporal standard deviation
    projection_dim:
        Optional per-level projection dimension. If provided, every pooled level
        vector is passed through its own ``LazyLinear(projection_dim)`` before
        aggregation. This is useful when you want to keep the final embedding
        size under control even when the separator has many channels.
    aggregation:
        How to aggregate multiple pooled levels.
        Supported values:
        - ``"concat"``: concatenate level embeddings
        - ``"mean"``: average them after optional projection
    dropout:
        Dropout probability applied after each optional projection and before
        the final aggregation.
    use_layernorm:
        Whether to apply ``LayerNorm`` to the final embedding.
    l2_normalize:
        Whether to L2-normalize the final embedding along the feature axis.
    detach_feature_maps:
        If True, detach feature maps before pooling. Useful when you only want
        the feature head for post-hoc embedding extraction without backprop.
    """

    pooling: str = "mean"
    projection_dim: Optional[int] = None
    aggregation: str = "concat"
    dropout: float = 0.0
    use_layernorm: bool = False
    l2_normalize: bool = False
    detach_feature_maps: bool = False


def _validate_pooling(pooling: str) -> str:
    pooling = pooling.lower()
    supported = {
        "mean",
        "max",
        "mean_max",
        "mean_std",
        "temporal_mean",
        "temporal_max",
        "temporal_mean_max",
        "temporal_mean_std",
    }
    if pooling not in supported:
        raise ValueError(f"Unsupported pooling={pooling!r}. Supported: {sorted(supported)}")
    return pooling


def _validate_aggregation(aggregation: str) -> str:
    aggregation = aggregation.lower()
    supported = {"concat", "mean"}
    if aggregation not in supported:
        raise ValueError(f"Unsupported aggregation={aggregation!r}. Supported: {sorted(supported)}")
    return aggregation


def _ensure_feature_map_list(features: Union[TensorOrTensors, SeparatorOutput]) -> List[torch.Tensor]:
    """Normalize various feature inputs into a list of 4D tensors.

    Accepted inputs
    ---------------
    1. A single 4D tensor ``[B,C,F,T]``
    2. A list/tuple of 4D tensors
    3. A separator output dictionary containing ``"asd_feature_maps"``
    """
    if isinstance(features, dict):
        if "asd_feature_maps" not in features:
            raise KeyError("Expected key 'asd_feature_maps' in separator output dictionary.")
        features = features["asd_feature_maps"]  # type: ignore[assignment]

    if torch.is_tensor(features):
        feature_list = [features]
    elif isinstance(features, (list, tuple)):
        feature_list = list(features)
    else:
        raise TypeError(
            "features must be a torch.Tensor, a sequence of tensors, or a separator output dict. "
            f"Got {type(features)!r}."
        )

    if len(feature_list) == 0:
        raise ValueError("Received an empty feature list.")

    validated: List[torch.Tensor] = []
    for i, feat in enumerate(feature_list):
        if not torch.is_tensor(feat):
            raise TypeError(f"Feature at index {i} is not a tensor: got {type(feat)!r}.")
        if feat.ndim != 4:
            raise ValueError(
                f"Each feature map must have shape [B,C,F,T]. Feature {i} has shape {tuple(feat.shape)}."
            )
        validated.append(feat)
    return validated


def pool_feature_map(feature_map: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """Pool a 2D feature map ``[B,C,F,T]`` into a vector ``[B,D]``.

    Parameters
    ----------
    feature_map:
        Input tensor with shape ``[B,C,F,T]``.
    pooling:
        Pooling mode; see :class:`FeatureHeadConfig`.

    Notes
    -----
    ``temporal_*`` pooling modes first aggregate the frequency axis with a mean
    reduction to obtain a sequence ``[B,C,T]``, then apply the requested
    temporal pooling over ``T``.
    """
    pooling = _validate_pooling(pooling)
    if feature_map.ndim != 4:
        raise ValueError(f"Expected [B,C,F,T], got {tuple(feature_map.shape)}")

    mean_vec = feature_map.mean(dim=(2, 3))

    if pooling == "mean":
        return mean_vec
    if pooling == "max":
        return feature_map.amax(dim=(2, 3))
    if pooling == "mean_max":
        max_vec = feature_map.amax(dim=(2, 3))
        return torch.cat([mean_vec, max_vec], dim=1)
    if pooling == "mean_std":
        flat = feature_map.flatten(start_dim=2)
        std_vec = flat.std(dim=2, unbiased=False)
        return torch.cat([mean_vec, std_vec], dim=1)

    temporal_seq = feature_map.mean(dim=2)  # [B,C,T]

    if pooling == "temporal_mean":
        return temporal_seq.mean(dim=2)
    if pooling == "temporal_max":
        return temporal_seq.amax(dim=2)
    if pooling == "temporal_mean_max":
        temporal_mean = temporal_seq.mean(dim=2)
        temporal_max = temporal_seq.amax(dim=2)
        return torch.cat([temporal_mean, temporal_max], dim=1)
    if pooling == "temporal_mean_std":
        temporal_mean = temporal_seq.mean(dim=2)
        temporal_std = temporal_seq.std(dim=2, unbiased=False)
        return torch.cat([temporal_mean, temporal_std], dim=1)

    raise RuntimeError(f"Unexpected pooling mode: {pooling!r}")


class SeparatorFeatureHead(nn.Module):
    """Convert separator feature maps into a fixed-dimensional embedding.

    Input
    -----
    One of the following:
    - a separator output dictionary containing ``"asd_feature_maps"``
    - a single feature map ``[B,C,F,T]``
    - a list/tuple of feature maps ``[B,C,F,T]``

    Output dictionary
    -----------------
    - ``embedding``: final embedding ``[B,D]``
    - ``level_embeddings``: pooled vectors for each level (after optional projection)
    - ``raw_level_embeddings``: pooled vectors before optional projection
    - ``num_levels``: number of feature levels
    - ``level_dims``: output dim of each level after pooling / projection
    - ``pooling``: pooling type used
    - ``aggregation``: aggregation type used
    """

    def __init__(self, config: Optional[FeatureHeadConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = FeatureHeadConfig(**kwargs)
        self.config = config
        self.pooling = _validate_pooling(config.pooling)
        self.aggregation = _validate_aggregation(config.aggregation)

        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0.0 else nn.Identity()
        self._level_projections = nn.ModuleList()
        self._final_norm: Optional[nn.LayerNorm] = None

    def _ensure_projection_layers(self, num_levels: int) -> None:
        if self.config.projection_dim is None:
            return
        while len(self._level_projections) < num_levels:
            self._level_projections.append(nn.LazyLinear(self.config.projection_dim))

    def _pool_levels(self, feature_maps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        pooled: List[torch.Tensor] = []
        for feat in feature_maps:
            if self.config.detach_feature_maps:
                feat = feat.detach()
            pooled.append(pool_feature_map(feat, pooling=self.pooling))
        return pooled

    def _project_levels(self, pooled_levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if self.config.projection_dim is None:
            return [self.dropout(vec) for vec in pooled_levels]

        self._ensure_projection_layers(len(pooled_levels))
        projected: List[torch.Tensor] = []
        for i, vec in enumerate(pooled_levels):
            projected_vec = self._level_projections[i](vec)
            projected_vec = self.dropout(projected_vec)
            projected.append(projected_vec)
        return projected

    def _aggregate_levels(self, level_embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(level_embeddings) == 0:
            raise ValueError("Cannot aggregate an empty list of level embeddings.")

        if self.aggregation == "concat":
            embedding = torch.cat(list(level_embeddings), dim=1)
        elif self.aggregation == "mean":
            base_dim = level_embeddings[0].shape[1]
            for i, vec in enumerate(level_embeddings):
                if vec.shape[1] != base_dim:
                    raise ValueError(
                        "aggregation='mean' requires equal dimensions for every level. "
                        f"Level 0 dim={base_dim}, level {i} dim={vec.shape[1]}. "
                        "Use projection_dim to force a common dimension."
                    )
            stacked = torch.stack(list(level_embeddings), dim=1)  # [B,L,D]
            embedding = stacked.mean(dim=1)
        else:
            raise RuntimeError(f"Unexpected aggregation mode: {self.aggregation!r}")

        if self.config.use_layernorm:
            if self._final_norm is None or self._final_norm.normalized_shape != (embedding.shape[1],):
                self._final_norm = nn.LayerNorm(embedding.shape[1]).to(embedding.device)
            embedding = self._final_norm(embedding)

        if self.config.l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(self, features: Union[TensorOrTensors, SeparatorOutput]) -> Dict[str, object]:
        feature_maps = _ensure_feature_map_list(features)
        raw_level_embeddings = self._pool_levels(feature_maps)
        level_embeddings = self._project_levels(raw_level_embeddings)
        embedding = self._aggregate_levels(level_embeddings)

        return {
            "embedding": embedding,
            "level_embeddings": level_embeddings,
            "raw_level_embeddings": raw_level_embeddings,
            "num_levels": len(level_embeddings),
            "level_dims": [int(vec.shape[1]) for vec in level_embeddings],
            "pooling": self.pooling,
            "aggregation": self.aggregation,
        }


def build_feature_head(**kwargs) -> SeparatorFeatureHead:
    """Convenience builder."""
    return SeparatorFeatureHead(**kwargs)


__all__ = [
    "FeatureHeadConfig",
    "pool_feature_map",
    "SeparatorFeatureHead",
    "build_feature_head",
]
