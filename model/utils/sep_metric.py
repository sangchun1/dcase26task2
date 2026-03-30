"""Evaluation metrics for the ASD source-separation proxy pipeline.

This module keeps *validation / monitoring metrics* separate from the training
losses in ``sep_loss.py``.

Main goals
----------
1. Compute practical separator metrics for validation:
   - SI-SDR
   - SI-SDR improvement (SI-SDRi)
   - spectrogram L1 / MSE / spectral convergence
   - optional mask error
2. Return metrics in a logging-friendly dictionary.
3. Provide a lightweight running tracker for epoch-level aggregation.

Recommended first usage
-----------------------
For the first separator MVP, the most useful validation metrics are:
- ``si_sdr_db``
- ``si_sdri_db``
- ``spec_l1``

Example
-------
>>> metrics = compute_separator_metrics(
...     pred_wave=pred_wave,
...     target_wave=target_wave,
...     mix_wave=mix_wave,
...     pred_spec=pred_spec,
...     target_spec=target_spec,
... )
>>> metrics["si_sdri_db"]

If you want to accumulate metrics across an epoch:
>>> tracker = SeparationMetricTracker()
>>> tracker.update(**metrics)
>>> epoch_metrics = tracker.compute()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
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
    raise ValueError(f"Expected spectrogram shape [B,F,T] or [B,1,F,T], got {tuple(spec.shape)}.")


# -----------------------------------------------------------------------------
# Waveform metrics
# -----------------------------------------------------------------------------

def si_sdr_db(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    zero_mean: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute batch SI-SDR in dB.

    Returns
    -------
    torch.Tensor
        Shape ``[B]``.
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
    target_energy = torch.sum(target_wave**2, dim=1, keepdim=True).clamp_min(eps)
    s_target = dot * target_wave / target_energy
    e_noise = pred_wave - s_target

    target_power = torch.sum(s_target**2, dim=1).clamp_min(eps)
    noise_power = torch.sum(e_noise**2, dim=1).clamp_min(eps)
    return 10.0 * torch.log10(target_power / noise_power)



def si_sdri_db(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    mix_wave: torch.Tensor,
    zero_mean: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute SI-SDR improvement (SI-SDRi) in dB.

    Defined as:
        SI-SDR(pred, target) - SI-SDR(mixture, target)
    """
    pred = si_sdr_db(pred_wave, target_wave, zero_mean=zero_mean, eps=eps)
    baseline = si_sdr_db(mix_wave, target_wave, zero_mean=zero_mean, eps=eps)
    return pred - baseline



def snr_db(
    estimate_wave: torch.Tensor,
    reference_wave: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute simple reconstruction SNR in dB.

    This is *not* scale-invariant, so SI-SDR should generally be preferred.
    It is still useful as an additional debugging metric.
    """
    estimate_wave = _ensure_wave_batch(estimate_wave)
    reference_wave = _ensure_wave_batch(reference_wave)

    if estimate_wave.shape != reference_wave.shape:
        raise ValueError(
            f"estimate_wave and reference_wave must have the same shape; got "
            f"{tuple(estimate_wave.shape)} vs {tuple(reference_wave.shape)}."
        )

    noise = estimate_wave - reference_wave
    signal_power = torch.sum(reference_wave**2, dim=1).clamp_min(eps)
    noise_power = torch.sum(noise**2, dim=1).clamp_min(eps)
    return 10.0 * torch.log10(signal_power / noise_power)


# -----------------------------------------------------------------------------
# Spectrogram / mask metrics
# -----------------------------------------------------------------------------

def spectrogram_l1_metric(pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    """Per-batch mean L1 spectrogram error."""
    pred_spec = _ensure_spec_batch(pred_spec)
    target_spec = _ensure_spec_batch(target_spec)
    return F.l1_loss(pred_spec, target_spec, reduction="mean")



def spectrogram_mse_metric(pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    """Per-batch mean MSE spectrogram error."""
    pred_spec = _ensure_spec_batch(pred_spec)
    target_spec = _ensure_spec_batch(target_spec)
    return F.mse_loss(pred_spec, target_spec, reduction="mean")



def spectral_convergence_metric(
    pred_spec: torch.Tensor,
    target_spec: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Spectral convergence metric.

    Computes, per sample,
        ||target - pred||_F / ||target||_F
    and averages across the batch.
    """
    pred_spec = _ensure_spec_batch(pred_spec)
    target_spec = _ensure_spec_batch(target_spec)

    diff = (target_spec - pred_spec).flatten(start_dim=1)
    target = target_spec.flatten(start_dim=1)

    num = torch.linalg.norm(diff, ord=2, dim=1)
    den = torch.linalg.norm(target, ord=2, dim=1).clamp_min(eps)
    return (num / den).mean()



def mask_l1_metric(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean L1 error between predicted and target masks."""
    pred_mask = _ensure_spec_batch(pred_mask)
    target_mask = _ensure_spec_batch(target_mask)
    return F.l1_loss(pred_mask, target_mask, reduction="mean")


# -----------------------------------------------------------------------------
# Config + high-level metric computation
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SeparationMetricConfig:
    """Configuration for :func:`compute_separator_metrics`.

    Parameters
    ----------
    compute_sisdr:
        Whether to compute SI-SDR.
    compute_sisdri:
        Whether to compute SI-SDRi. Requires ``mix_wave``.
    compute_snr:
        Whether to compute the simple non-scale-invariant SNR.
    compute_spec_l1:
        Whether to compute spectrogram L1.
    compute_spec_mse:
        Whether to compute spectrogram MSE.
    compute_spectral_convergence:
        Whether to compute spectral convergence.
    compute_mask_l1:
        Whether to compute mask L1 when masks are provided.
    zero_mean_sisdr:
        Whether to remove mean before SI-SDR / SI-SDRi.
    detach:
        Whether to detach all returned metric tensors.
    """

    compute_sisdr: bool = True
    compute_sisdri: bool = True
    compute_snr: bool = False
    compute_spec_l1: bool = True
    compute_spec_mse: bool = False
    compute_spectral_convergence: bool = False
    compute_mask_l1: bool = False
    zero_mean_sisdr: bool = True
    detach: bool = True



def _maybe_detach(x: torch.Tensor, detach: bool) -> torch.Tensor:
    return x.detach() if detach else x



def compute_separator_metrics(
    *,
    pred_wave: Optional[torch.Tensor] = None,
    target_wave: Optional[torch.Tensor] = None,
    mix_wave: Optional[torch.Tensor] = None,
    pred_spec: Optional[torch.Tensor] = None,
    target_spec: Optional[torch.Tensor] = None,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    config: Optional[SeparationMetricConfig] = None,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """Compute a validation-metric dictionary for separator outputs.

    Parameters are keyword-only on purpose so call sites remain readable.

    Returns
    -------
    Dict[str, torch.Tensor]
        Scalar tensors suitable for logging.
    """
    cfg = SeparationMetricConfig() if config is None else config
    metrics: Dict[str, torch.Tensor] = {}

    def _name(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    # Waveform metrics ---------------------------------------------------------
    if cfg.compute_sisdr:
        if pred_wave is None or target_wave is None:
            raise ValueError("pred_wave and target_wave are required to compute SI-SDR.")
        value = si_sdr_db(pred_wave, target_wave, zero_mean=cfg.zero_mean_sisdr).mean()
        metrics[_name("si_sdr_db")] = _maybe_detach(value, cfg.detach)

    if cfg.compute_sisdri:
        if pred_wave is None or target_wave is None or mix_wave is None:
            raise ValueError("pred_wave, target_wave, and mix_wave are required to compute SI-SDRi.")
        value = si_sdri_db(
            pred_wave,
            target_wave,
            mix_wave,
            zero_mean=cfg.zero_mean_sisdr,
        ).mean()
        metrics[_name("si_sdri_db")] = _maybe_detach(value, cfg.detach)

    if cfg.compute_snr:
        if pred_wave is None or target_wave is None:
            raise ValueError("pred_wave and target_wave are required to compute SNR.")
        value = snr_db(pred_wave, target_wave).mean()
        metrics[_name("snr_db")] = _maybe_detach(value, cfg.detach)

    # Spectrogram metrics ------------------------------------------------------
    if cfg.compute_spec_l1:
        if pred_spec is None or target_spec is None:
            raise ValueError("pred_spec and target_spec are required to compute spectrogram L1.")
        value = spectrogram_l1_metric(pred_spec, target_spec)
        metrics[_name("spec_l1")] = _maybe_detach(value, cfg.detach)

    if cfg.compute_spec_mse:
        if pred_spec is None or target_spec is None:
            raise ValueError("pred_spec and target_spec are required to compute spectrogram MSE.")
        value = spectrogram_mse_metric(pred_spec, target_spec)
        metrics[_name("spec_mse")] = _maybe_detach(value, cfg.detach)

    if cfg.compute_spectral_convergence:
        if pred_spec is None or target_spec is None:
            raise ValueError(
                "pred_spec and target_spec are required to compute spectral convergence."
            )
        value = spectral_convergence_metric(pred_spec, target_spec)
        metrics[_name("spectral_convergence")] = _maybe_detach(value, cfg.detach)

    # Mask metrics -------------------------------------------------------------
    if cfg.compute_mask_l1:
        if pred_mask is None or target_mask is None:
            raise ValueError("pred_mask and target_mask are required to compute mask L1.")
        value = mask_l1_metric(pred_mask, target_mask)
        metrics[_name("mask_l1")] = _maybe_detach(value, cfg.detach)

    return metrics


# -----------------------------------------------------------------------------
# Running tracker for epoch aggregation
# -----------------------------------------------------------------------------
class SeparationMetricTracker:
    """Tiny running-average tracker for separator validation metrics.

    This is intentionally lightweight so it can be used either inside a
    LightningModule or in standalone validation scripts.

    Example
    -------
    >>> tracker = SeparationMetricTracker()
    >>> tracker.update(si_sdr_db=torch.tensor(8.1), si_sdri_db=torch.tensor(3.2))
    >>> tracker.update(si_sdr_db=torch.tensor(9.1), si_sdri_db=torch.tensor(4.0))
    >>> tracker.compute()
    {'si_sdr_db': tensor(8.6), 'si_sdri_db': tensor(3.6)}
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, torch.Tensor] = {}
        self._count: Dict[str, int] = {}

    def update(self, **metrics: torch.Tensor) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.tensor(float(value))
            value = value.detach().float().cpu()
            if value.ndim != 0:
                value = value.mean()
            if key not in self._sum:
                self._sum[key] = value.clone()
                self._count[key] = 1
            else:
                self._sum[key] += value
                self._count[key] += 1

    def update_from_mapping(self, metrics: Mapping[str, torch.Tensor]) -> None:
        self.update(**dict(metrics))

    def compute(self) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for key, value in self._sum.items():
            count = max(self._count.get(key, 0), 1)
            result[key] = value / float(count)
        return result

    def state_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            "sum": {k: v.clone() for k, v in self._sum.items()},
            "count": dict(self._count),
        }

    def load_state_dict(self, state_dict: Mapping[str, Mapping[str, object]]) -> None:
        self._sum = {k: v.clone() for k, v in state_dict["sum"].items()}  # type: ignore[index]
        self._count = {k: int(v) for k, v in state_dict["count"].items()}  # type: ignore[index]


# -----------------------------------------------------------------------------
# Practical default helper
# -----------------------------------------------------------------------------

def build_default_separator_metric_config() -> SeparationMetricConfig:
    """Return the recommended first validation setup.

    Default metrics:
    - SI-SDR
    - SI-SDRi
    - spectrogram L1
    """
    return SeparationMetricConfig(
        compute_sisdr=True,
        compute_sisdri=True,
        compute_snr=False,
        compute_spec_l1=True,
        compute_spec_mse=False,
        compute_spectral_convergence=False,
        compute_mask_l1=False,
        zero_mean_sisdr=True,
        detach=True,
    )


__all__ = [
    "SeparationMetricConfig",
    "SeparationMetricTracker",
    "build_default_separator_metric_config",
    "compute_separator_metrics",
    "mask_l1_metric",
    "si_sdr_db",
    "si_sdri_db",
    "snr_db",
    "spectral_convergence_metric",
    "spectrogram_l1_metric",
    "spectrogram_mse_metric",
]
