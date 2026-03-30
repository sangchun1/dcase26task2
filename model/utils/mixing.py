"""Utilities for building synthetic source-separation mixtures.

This module is designed for the ASD source-separation proxy pipeline.
It focuses on one job: given a target machine waveform and an interference
waveform, create a stable training mixture with a desired SNR.

Main features
-------------
- zero-mean normalization
- mono conversion safety check
- crop / pad helpers
- random segment sampling
- interference scaling to a target SNR
- optional peak normalization after mixing
- metadata dictionary for logging/debugging

The functions here operate on NumPy arrays because the current project's data
loading stack already uses ``soundfile`` + NumPy before converting to Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray]
EPS = 1e-8


@dataclass(frozen=True)
class MixingConfig:
    """Configuration for synthetic mixture generation.

    Attributes
    ----------
    sample_rate:
        Audio sample rate.
    segment_seconds:
        Target segment duration in seconds.
    snr_min_db:
        Minimum SNR sampled during training.
    snr_max_db:
        Maximum SNR sampled during training.
    peak_normalize:
        Whether to peak-normalize mixture/target/interference together after
        mixing. This keeps values within a stable range.
    zero_mean:
        Whether to remove DC offset before mixing.
    """

    sample_rate: int = 16000
    segment_seconds: float = 2.0
    snr_min_db: float = -5.0
    snr_max_db: float = 5.0
    peak_normalize: bool = True
    zero_mean: bool = True

    @property
    def segment_samples(self) -> int:
        return int(round(self.sample_rate * self.segment_seconds))


def ensure_float32_mono(wave: ArrayLike) -> np.ndarray:
    """Convert input audio to a contiguous 1-D float32 mono waveform.

    Parameters
    ----------
    wave:
        Input waveform. Accepts shape ``[T]`` or ``[T, C]`` or ``[C, T]``.

    Returns
    -------
    np.ndarray
        Mono waveform of shape ``[T]`` and dtype ``float32``.
    """

    wave = np.asarray(wave, dtype=np.float32)

    if wave.ndim == 1:
        return np.ascontiguousarray(wave)

    if wave.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D waveform, got shape {wave.shape}.")

    # Handle [T, C] or [C, T]
    if wave.shape[0] == 1:
        mono = wave[0]
    elif wave.shape[1] == 1:
        mono = wave[:, 0]
    elif wave.shape[0] < wave.shape[1]:
        # Likely [C, T]
        mono = wave.mean(axis=0)
    else:
        # Likely [T, C]
        mono = wave.mean(axis=1)

    return np.ascontiguousarray(mono.astype(np.float32, copy=False))


def zero_mean(wave: ArrayLike) -> np.ndarray:
    """Remove DC offset from a waveform."""
    wave = ensure_float32_mono(wave)
    return wave - np.mean(wave, dtype=np.float32)


def rms(wave: ArrayLike, eps: float = EPS) -> float:
    """Compute RMS energy."""
    wave = ensure_float32_mono(wave)
    return float(np.sqrt(np.mean(np.square(wave), dtype=np.float32) + eps))


def peak_normalize(wave: ArrayLike, eps: float = EPS) -> np.ndarray:
    """Normalize waveform by its absolute peak."""
    wave = ensure_float32_mono(wave)
    peak = float(np.max(np.abs(wave)))
    if peak < eps:
        return wave.copy()
    return wave / peak


def match_length(wave: ArrayLike, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad or crop a waveform to a fixed number of samples.

    This uses deterministic center crop for overlong signals. For random crop,
    use :func:`random_segment`.
    """
    wave = ensure_float32_mono(wave)
    length = wave.shape[0]

    if length == target_length:
        return wave.copy()
    if length < target_length:
        pad = target_length - length
        return np.pad(wave, (0, pad), mode="constant", constant_values=pad_value).astype(np.float32)

    start = (length - target_length) // 2
    return wave[start : start + target_length].astype(np.float32, copy=False)


def random_segment(
    wave: ArrayLike,
    segment_length: int,
    rng: Optional[np.random.Generator] = None,
    pad_if_short: bool = True,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Sample a random fixed-length segment from a waveform.

    Parameters
    ----------
    wave:
        Input waveform.
    segment_length:
        Desired output length in samples.
    rng:
        Optional NumPy random generator.
    pad_if_short:
        If True and the waveform is shorter than ``segment_length``, pad with
        zeros. Otherwise raise ``ValueError``.
    pad_value:
        Constant padding value when padding is used.
    """
    wave = ensure_float32_mono(wave)
    rng = rng or np.random.default_rng()
    length = wave.shape[0]

    if length == segment_length:
        return wave.copy()

    if length < segment_length:
        if not pad_if_short:
            raise ValueError(
                f"Waveform too short: length={length}, required={segment_length}."
            )
        pad = segment_length - length
        return np.pad(wave, (0, pad), mode="constant", constant_values=pad_value).astype(np.float32)

    start = int(rng.integers(0, length - segment_length + 1))
    return wave[start : start + segment_length].astype(np.float32, copy=False)


def sample_snr_db(
    snr_min_db: float,
    snr_max_db: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sample an SNR value uniformly from a range."""
    rng = rng or np.random.default_rng()
    return float(rng.uniform(snr_min_db, snr_max_db))


def scale_interference_to_snr(
    target: ArrayLike,
    interference: ArrayLike,
    snr_db: float,
    eps: float = EPS,
) -> np.ndarray:
    """Scale interference so that target/interference reaches the desired SNR.

    SNR definition used here:
        snr_db = 20 * log10(rms(target) / rms(interference_scaled))
    """
    target = ensure_float32_mono(target)
    interference = ensure_float32_mono(interference)

    target_rms = rms(target, eps=eps)
    interference_rms = rms(interference, eps=eps)

    desired_interference_rms = target_rms / (10.0 ** (snr_db / 20.0))
    scale = desired_interference_rms / max(interference_rms, eps)
    return (interference * scale).astype(np.float32, copy=False)


def mix_signals(
    target: ArrayLike,
    interference: ArrayLike,
    snr_db: float,
    peak_norm: bool = True,
    zero_mean_before_mix: bool = True,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a mixture and return aligned target/interference copies.

    Returns
    -------
    mixture, target_out, interference_out : np.ndarray
        All arrays have identical length and dtype ``float32``.
    """
    target = ensure_float32_mono(target)
    interference = ensure_float32_mono(interference)

    if target.shape[0] != interference.shape[0]:
        raise ValueError(
            "Target and interference must have the same length before mixing. "
            f"Got {target.shape[0]} and {interference.shape[0]}."
        )

    if zero_mean_before_mix:
        target = zero_mean(target)
        interference = zero_mean(interference)

    interference_scaled = scale_interference_to_snr(target, interference, snr_db, eps=eps)
    mixture = target + interference_scaled

    if peak_norm:
        peak = float(np.max(np.abs([mixture, target, interference_scaled])))
        if peak > eps:
            mixture = mixture / peak
            target = target / peak
            interference_scaled = interference_scaled / peak

    return (
        np.ascontiguousarray(mixture.astype(np.float32, copy=False)),
        np.ascontiguousarray(target.astype(np.float32, copy=False)),
        np.ascontiguousarray(interference_scaled.astype(np.float32, copy=False)),
    )


def build_training_mixture(
    target_wave: ArrayLike,
    interference_wave: ArrayLike,
    config: Optional[MixingConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """Build one synthetic training sample for source-separation proxy learning.

    Workflow
    --------
    1. convert both signals to mono float32
    2. random-crop/pad each to a fixed segment length
    3. sample SNR from the configured range
    4. scale interference to target SNR and mix
    5. optionally peak-normalize outputs

    Returns
    -------
    dict
        Keys:
        - ``mix_wave``
        - ``target_wave``
        - ``interference_wave``
        - ``snr_db``
        - ``segment_length``
    """
    config = config or MixingConfig()
    rng = rng or np.random.default_rng()
    seg_len = config.segment_samples

    target_seg = random_segment(target_wave, seg_len, rng=rng, pad_if_short=True)
    interference_seg = random_segment(interference_wave, seg_len, rng=rng, pad_if_short=True)
    snr_db = sample_snr_db(config.snr_min_db, config.snr_max_db, rng=rng)

    mix_wave, target_seg, interference_seg = mix_signals(
        target=target_seg,
        interference=interference_seg,
        snr_db=snr_db,
        peak_norm=config.peak_normalize,
        zero_mean_before_mix=config.zero_mean,
    )

    return {
        "mix_wave": mix_wave,
        "target_wave": target_seg,
        "interference_wave": interference_seg,
        "snr_db": snr_db,
        "segment_length": seg_len,
    }


def build_fixed_snr_mixture(
    target_wave: ArrayLike,
    interference_wave: ArrayLike,
    sample_rate: int = 16000,
    segment_seconds: float = 2.0,
    snr_db: float = 0.0,
    peak_norm: bool = True,
    zero_mean_before_mix: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """Build a mixture at a specified SNR.

    Useful for validation, ablation, and SI-SDRi evaluation at fixed SNR values
    such as -5, 0, and 5 dB.
    """
    rng = rng or np.random.default_rng()
    seg_len = int(round(sample_rate * segment_seconds))

    target_seg = random_segment(target_wave, seg_len, rng=rng, pad_if_short=True)
    interference_seg = random_segment(interference_wave, seg_len, rng=rng, pad_if_short=True)

    mix_wave, target_seg, interference_seg = mix_signals(
        target=target_seg,
        interference=interference_seg,
        snr_db=snr_db,
        peak_norm=peak_norm,
        zero_mean_before_mix=zero_mean_before_mix,
    )

    return {
        "mix_wave": mix_wave,
        "target_wave": target_seg,
        "interference_wave": interference_seg,
        "snr_db": float(snr_db),
        "segment_length": seg_len,
    }


def estimate_realized_snr_db(
    target_wave: ArrayLike,
    interference_wave: ArrayLike,
    eps: float = EPS,
) -> float:
    """Estimate realized SNR after mixing/normalization for sanity checks."""
    target_wave = ensure_float32_mono(target_wave)
    interference_wave = ensure_float32_mono(interference_wave)
    return float(20.0 * np.log10((rms(target_wave, eps) + eps) / (rms(interference_wave, eps) + eps)))


__all__ = [
    "MixingConfig",
    "build_fixed_snr_mixture",
    "build_training_mixture",
    "ensure_float32_mono",
    "estimate_realized_snr_db",
    "match_length",
    "mix_signals",
    "peak_normalize",
    "random_segment",
    "rms",
    "sample_snr_db",
    "scale_interference_to_snr",
    "zero_mean",
]
