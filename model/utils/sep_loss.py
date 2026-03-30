"""Loss functions for the ASD source-separation proxy pipeline.

This module provides practical training losses for the first separator MVP:

- spectrogram-domain losses (L1 / MSE / spectral convergence)
- waveform-domain SI-SDR loss
- optional mask-domain supervision for debug / auxiliary training
- a single nn.Module that combines multiple terms with configurable weights

Recommended first setup
-----------------------
Use the separator with:
- input : mixture magnitude spectrogram
- target: target magnitude spectrogram
- output: predicted target magnitude or mask
- loss  : L1(spec) + SI-SDR(wave)

Typical usage
-------------
>>> loss_fn = SeparationLoss(l1_spec_weight=1.0, sisdr_weight=0.1)
>>> out = separator(mix_spec)
>>> pred_wave = frontend.pred_spec_to_wave(out["pred_spec"], mix_aux)
>>> losses = loss_fn(
...     pred_spec=out["pred_spec"],
...     target_spec=target_spec,
...     pred_wave=pred_wave,
...     target_wave=target_wave,
... )
>>> loss = losses["loss"]

The returned dictionary keeps each component separate so you can log them in a
LightningModule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# -----------------------------------------------------------------------------
# Shape helpers
# -----------------------------------------------------------------------------
def _ensure_wave_batch(wave: torch.Tensor) -> torch.Tensor:
    """Convert a waveform tensor to shape ``[B, T]``.

    Accepted shapes
    ---------------
    - ``[T]``
    - ``[B, T]``
    - ``[B, 1, T]``
    """
    if not torch.is_tensor(wave):
        raise TypeError(f"Expected torch.Tensor, got {type(wave)!r}.")

    if wave.ndim == 1:
        return wave.unsqueeze(0)
    if wave.ndim == 2:
        return wave
    if wave.ndim == 3 and wave.shape[1] == 1:
        return wave[:, 0, :]
    raise ValueError(f"Expected waveform shape [T], [B,T], or [B,1,T], got {tuple(wave.shape)}.")


def _ensure_spec_batch(spec: torch.Tensor) -> torch.Tensor:
    """Convert a spectrogram tensor to shape ``[B, 1, F, T]``."""
    if not torch.is_tensor(spec):
        raise TypeError(f"Expected torch.Tensor, got {type(spec)!r}.")

    if spec.ndim == 3:
        return spec.unsqueeze(1)
    if spec.ndim == 4 and spec.shape[1] == 1:
        return spec
    raise ValueError(
        f"Expected spectrogram shape [B,F,T] or [B,1,F,T], got {tuple(spec.shape)}."
    )


# -----------------------------------------------------------------------------
# Basic loss components
# -----------------------------------------------------------------------------
def l1_spectrogram_loss(
    pred_spec: torch.Tensor,
    target_spec: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss on spectrogram representations.

    Both tensors are interpreted as already being in the same representation,
    e.g. magnitude or log-magnitude.
    """
    pred_spec = _ensure_spec_batch(pred_spec)
    target_spec = _ensure_spec_batch(target_spec)
    return F.l1_loss(pred_spec, target_spec, reduction=reduction)


def mse_spectrogram_loss(
    pred_spec: torch.Tensor,
    target_spec: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE loss on spectrogram representations."""
    pred_spec = _ensure_spec_batch(pred_spec)
    target_spec = _ensure_spec_batch(target_spec)
    return F.mse_loss(pred_spec, target_spec, reduction=reduction)


def spectral_convergence_loss(
    pred_spec: torch.Tensor,
    target_spec: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Spectral convergence loss.

    Computes, per sample,
        ||target - pred||_F / ||target||_F
    and averages across the batch.

    This is commonly used in spectrogram reconstruction because it emphasizes
    relative structure rather than only absolute magnitude errors.
    """
    pred_spec = _ensure_spec_batch(pred_spec)
    target_spec = _ensure_spec_batch(target_spec)

    diff = (target_spec - pred_spec).flatten(start_dim=1)
    target = target_spec.flatten(start_dim=1)

    num = torch.linalg.norm(diff, ord=2, dim=1)
    den = torch.linalg.norm(target, ord=2, dim=1).clamp_min(eps)
    return (num / den).mean()


def mask_l1_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss on ratio masks."""
    pred_mask = _ensure_spec_batch(pred_mask)
    target_mask = _ensure_spec_batch(target_mask)
    return F.l1_loss(pred_mask, target_mask, reduction=reduction)


def mask_bce_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    clamp_target: bool = True,
) -> torch.Tensor:
    """Binary cross-entropy loss on mask predictions.

    This is mainly useful when the separator uses ``output_mode='mask'`` and an
    oracle mask is available from the frontend.
    """
    pred_mask = _ensure_spec_batch(pred_mask)
    target_mask = _ensure_spec_batch(target_mask)
    if clamp_target:
        target_mask = target_mask.clamp(0.0, 1.0)
    pred_mask = pred_mask.clamp(EPS, 1.0 - EPS)
    return F.binary_cross_entropy(pred_mask, target_mask)


# -----------------------------------------------------------------------------
# SI-SDR
# -----------------------------------------------------------------------------
def si_sdr(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    zero_mean: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute batch SI-SDR in dB.

    Returns a tensor of shape ``[B]``.

    Formula
    -------
    Let ``s`` be the target and ``x`` the estimate.
    The target projection is
        s_target = <x, s> / ||s||^2 * s
    and the residual is
        e_noise = x - s_target
    Then
        SI-SDR = 10 log10( ||s_target||^2 / ||e_noise||^2 )
    """
    pred_wave = _ensure_wave_batch(pred_wave)
    target_wave = _ensure_wave_batch(target_wave)

    if pred_wave.shape != target_wave.shape:
        raise ValueError(
            f"pred_wave and target_wave must have the same shape; got "
            f"{tuple(pred_wave.shape)} vs {tuple(target_wave.shape)}."
        )

    if zero_mean:
        pred_wave = pred_wave - pred_wave.mean(dim=1, keepdim=True)
        target_wave = target_wave - target_wave.mean(dim=1, keepdim=True)

    dot = torch.sum(pred_wave * target_wave, dim=1, keepdim=True)
    target_energy = torch.sum(target_wave ** 2, dim=1, keepdim=True).clamp_min(eps)
    s_target = dot * target_wave / target_energy
    e_noise = pred_wave - s_target

    target_power = torch.sum(s_target ** 2, dim=1).clamp_min(eps)
    noise_power = torch.sum(e_noise ** 2, dim=1).clamp_min(eps)
    return 10.0 * torch.log10(target_power / noise_power)


def neg_si_sdr_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    zero_mean: bool = True,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """Negative SI-SDR loss.

    Supported reductions: ``mean``, ``sum``, ``none``.
    """
    values = -si_sdr(pred_wave, target_wave, zero_mean=zero_mean, eps=eps)
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction={reduction!r}.")


# -----------------------------------------------------------------------------
# Combined loss module
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SeparationLossConfig:
    """Configuration for :class:`SeparationLoss`.

    Parameters
    ----------
    l1_spec_weight:
        Weight for L1 spectrogram reconstruction loss.
    mse_spec_weight:
        Weight for MSE spectrogram reconstruction loss.
    spectral_convergence_weight:
        Weight for spectral convergence loss.
    sisdr_weight:
        Weight for negative SI-SDR waveform loss.
    mask_l1_weight:
        Weight for mask-domain L1 loss.
    mask_bce_weight:
        Weight for mask-domain BCE loss.
    zero_mean_waveform:
        Whether SI-SDR uses zero-mean normalization.
    reduction:
        Reduction for L1/MSE/SI-SDR terms. For the combined module the total
        loss is always returned as a scalar.
    """

    l1_spec_weight: float = 1.0
    mse_spec_weight: float = 0.0
    spectral_convergence_weight: float = 0.0
    sisdr_weight: float = 0.1
    mask_l1_weight: float = 0.0
    mask_bce_weight: float = 0.0
    zero_mean_waveform: bool = True
    reduction: str = "mean"


class SeparationLoss(nn.Module):
    """Weighted combination of separator training losses.

    Expected inputs
    ---------------
    - ``pred_spec`` and ``target_spec`` for spectrogram-domain losses
    - ``pred_wave`` and ``target_wave`` for waveform-domain SI-SDR
    - optionally ``pred_mask`` and ``target_mask`` for mask supervision

    The forward method returns a dictionary with individual components and the
    final scalar under the key ``"loss"``.
    """

    def __init__(self, config: Optional[SeparationLossConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = SeparationLossConfig(**kwargs)
        self.config = config

    def forward(
        self,
        *,
        pred_spec: Optional[torch.Tensor] = None,
        target_spec: Optional[torch.Tensor] = None,
        pred_wave: Optional[torch.Tensor] = None,
        target_wave: Optional[torch.Tensor] = None,
        pred_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total_loss: Optional[torch.Tensor] = None

        def _add(name: str, value: torch.Tensor, weight: float) -> None:
            nonlocal total_loss
            weighted = value * float(weight)
            losses[name] = value
            losses[f"weighted_{name}"] = weighted
            total_loss = weighted if total_loss is None else total_loss + weighted

        if self.config.l1_spec_weight > 0.0:
            if pred_spec is None or target_spec is None:
                raise ValueError("l1_spec_weight > 0 requires pred_spec and target_spec.")
            value = l1_spectrogram_loss(pred_spec, target_spec, reduction=self.config.reduction)
            _add("l1_spec", value, self.config.l1_spec_weight)

        if self.config.mse_spec_weight > 0.0:
            if pred_spec is None or target_spec is None:
                raise ValueError("mse_spec_weight > 0 requires pred_spec and target_spec.")
            value = mse_spectrogram_loss(pred_spec, target_spec, reduction=self.config.reduction)
            _add("mse_spec", value, self.config.mse_spec_weight)

        if self.config.spectral_convergence_weight > 0.0:
            if pred_spec is None or target_spec is None:
                raise ValueError(
                    "spectral_convergence_weight > 0 requires pred_spec and target_spec."
                )
            value = spectral_convergence_loss(pred_spec, target_spec)
            _add("spectral_convergence", value, self.config.spectral_convergence_weight)

        if self.config.sisdr_weight > 0.0:
            if pred_wave is None or target_wave is None:
                raise ValueError("sisdr_weight > 0 requires pred_wave and target_wave.")
            value = neg_si_sdr_loss(
                pred_wave,
                target_wave,
                zero_mean=self.config.zero_mean_waveform,
                reduction=self.config.reduction,
            )
            _add("neg_si_sdr", value, self.config.sisdr_weight)

        if self.config.mask_l1_weight > 0.0:
            if pred_mask is None or target_mask is None:
                raise ValueError("mask_l1_weight > 0 requires pred_mask and target_mask.")
            value = mask_l1_loss(pred_mask, target_mask, reduction=self.config.reduction)
            _add("mask_l1", value, self.config.mask_l1_weight)

        if self.config.mask_bce_weight > 0.0:
            if pred_mask is None or target_mask is None:
                raise ValueError("mask_bce_weight > 0 requires pred_mask and target_mask.")
            value = mask_bce_loss(pred_mask, target_mask)
            _add("mask_bce", value, self.config.mask_bce_weight)

        if total_loss is None:
            raise ValueError(
                "No active loss terms. Set at least one non-zero weight in SeparationLossConfig."
            )

        losses["loss"] = total_loss
        return losses


def build_default_separator_loss() -> SeparationLoss:
    """Return the recommended MVP loss: ``L1(spec) + 0.1 * neg_SI-SDR``."""
    return SeparationLoss(
        l1_spec_weight=1.0,
        mse_spec_weight=0.0,
        spectral_convergence_weight=0.0,
        sisdr_weight=0.1,
        mask_l1_weight=0.0,
        mask_bce_weight=0.0,
    )


__all__ = [
    "SeparationLossConfig",
    "SeparationLoss",
    "l1_spectrogram_loss",
    "mse_spectrogram_loss",
    "spectral_convergence_loss",
    "mask_l1_loss",
    "mask_bce_loss",
    "si_sdr",
    "neg_si_sdr_loss",
    "build_default_separator_loss",
]
