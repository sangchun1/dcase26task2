"""Utilities for building synthetic source-separation mixtures.

This module is designed for the ASD source-separation proxy pipeline.
It keeps the original simple target+interference mixing workflow, but adds
optional augmentation hooks that are useful when porting ideas from stronger
source-separation systems:

- optional room impulse response (RIR) convolution
- optional background-noise injection
- richer metadata for debugging and ablation
- compatibility with both ``build_training_mixture`` and
  ``build_fixed_snr_mixture(..., config=...)`` call styles

The functions operate on NumPy arrays because the current project's data
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

    Parameters
    ----------
    sample_rate:
        Audio sample rate.
    segment_seconds:
        Target segment duration in seconds.
    snr_min_db, snr_max_db:
        Uniform sampling range for the main target-vs-interference SNR.
    peak_normalize:
        Whether to peak-normalize the final outputs together.
    zero_mean:
        Whether to remove DC offset before mixing.
    peak_target:
        Peak value after peak normalization.
    rir_probability:
        Probability of applying a provided shared/per-component RIR.
    normalize_rir:
        Whether to L1-normalize the RIR before convolution.
    apply_rir_to_target, apply_rir_to_interference, apply_rir_to_background:
        Component-wise switches for RIR augmentation.
    background_probability:
        Probability of using provided background noise during training.
    background_snr_min_db, background_snr_max_db:
        SNR range for background noise relative to the target.
    """

    sample_rate: int = 16000
    segment_seconds: float = 2.0
    snr_min_db: float = -5.0
    snr_max_db: float = 5.0
    peak_normalize: bool = True
    zero_mean: bool = True
    peak_target: float = 0.99

    rir_probability: float = 0.0
    normalize_rir: bool = True
    apply_rir_to_target: bool = True
    apply_rir_to_interference: bool = True
    apply_rir_to_background: bool = True

    background_probability: float = 0.0
    background_snr_min_db: Optional[float] = None
    background_snr_max_db: Optional[float] = None

    @property
    def segment_samples(self) -> int:
        return int(round(self.sample_rate * self.segment_seconds))


def ensure_float32_mono(wave: ArrayLike) -> np.ndarray:
    """Convert input audio to a contiguous 1-D float32 mono waveform."""
    wave = np.asarray(wave, dtype=np.float32)

    if wave.ndim == 1:
        return np.ascontiguousarray(wave)

    if wave.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D waveform, got shape {wave.shape}.")

    if wave.shape[0] == 1:
        mono = wave[0]
    elif wave.shape[1] == 1:
        mono = wave[:, 0]
    elif wave.shape[0] < wave.shape[1]:
        mono = wave.mean(axis=0)  # likely [C, T]
    else:
        mono = wave.mean(axis=1)  # likely [T, C]

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
    """Pad or crop a waveform to a fixed number of samples."""
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
    """Sample a random fixed-length segment from a waveform."""
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


def should_apply(probability: float, rng: Optional[np.random.Generator] = None) -> bool:
    """Bernoulli helper for optional augmentations."""
    rng = rng or np.random.default_rng()
    probability = float(np.clip(probability, 0.0, 1.0))
    return bool(rng.random() < probability)


def scale_signal_to_reference_snr(
    reference: ArrayLike,
    signal: ArrayLike,
    snr_db: float,
    eps: float = EPS,
) -> np.ndarray:
    """Scale ``signal`` so that ``reference / signal`` matches ``snr_db``."""
    reference = ensure_float32_mono(reference)
    signal = ensure_float32_mono(signal)

    reference_rms = rms(reference, eps=eps)
    signal_rms = rms(signal, eps=eps)

    desired_signal_rms = reference_rms / (10.0 ** (snr_db / 20.0))
    scale = desired_signal_rms / max(signal_rms, eps)
    return (signal * scale).astype(np.float32, copy=False)


def scale_interference_to_snr(
    target: ArrayLike,
    interference: ArrayLike,
    snr_db: float,
    eps: float = EPS,
) -> np.ndarray:
    """Backward-compatible alias for target-vs-interference scaling."""
    return scale_signal_to_reference_snr(target, interference, snr_db, eps=eps)


def normalize_rir_kernel(rir: ArrayLike, eps: float = EPS) -> np.ndarray:
    """L1-normalize an RIR kernel for stable convolution energy."""
    rir = ensure_float32_mono(rir)
    norm = float(np.sum(np.abs(rir), dtype=np.float32))
    if norm < eps:
        return rir.copy()
    return np.ascontiguousarray((rir / norm).astype(np.float32, copy=False))


def convolve_with_rir(
    wave: ArrayLike,
    rir: ArrayLike,
    normalize_kernel: bool = True,
    keep_length: bool = True,
) -> np.ndarray:
    """Apply 1-D convolution with an RIR and optionally keep original length."""
    wave = ensure_float32_mono(wave)
    rir = ensure_float32_mono(rir)

    if normalize_kernel:
        rir = normalize_rir_kernel(rir)

    convolved = np.convolve(wave, rir, mode="full").astype(np.float32, copy=False)

    if not keep_length:
        return np.ascontiguousarray(convolved)

    if convolved.shape[0] < wave.shape[0]:
        return match_length(convolved, wave.shape[0], pad_value=0.0)
    return np.ascontiguousarray(convolved[: wave.shape[0]])


def maybe_apply_rir(
    wave: ArrayLike,
    rir_wave: Optional[ArrayLike],
    *,
    enabled: bool,
    probability: float,
    normalize_kernel: bool,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, bool]:
    """Optionally convolve a waveform with a provided RIR."""
    wave = ensure_float32_mono(wave)
    if not enabled or rir_wave is None:
        return wave, False

    if not should_apply(probability=probability, rng=rng):
        return wave, False

    return convolve_with_rir(wave, rir_wave, normalize_kernel=normalize_kernel), True


def _joint_peak_normalize(
    mixture: np.ndarray,
    target: np.ndarray,
    interference: np.ndarray,
    background: Optional[np.ndarray] = None,
    *,
    peak_target: float = 0.99,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    """Normalize all components with one shared factor."""
    candidates = [np.abs(mixture), np.abs(target), np.abs(interference)]
    if background is not None:
        candidates.append(np.abs(background))
    peak = float(max(float(np.max(x)) for x in candidates))
    if peak <= eps:
        return mixture, target, interference, background, 1.0

    scale = float(peak_target / peak)
    mixture = (mixture * scale).astype(np.float32, copy=False)
    target = (target * scale).astype(np.float32, copy=False)
    interference = (interference * scale).astype(np.float32, copy=False)
    if background is not None:
        background = (background * scale).astype(np.float32, copy=False)
    return mixture, target, interference, background, scale


def mix_signals(
    target: ArrayLike,
    interference: ArrayLike,
    snr_db: float,
    peak_norm: bool = True,
    zero_mean_before_mix: bool = True,
    eps: float = EPS,
    background: Optional[ArrayLike] = None,
    background_snr_db: Optional[float] = None,
    peak_target: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a mixture and return aligned target/interference copies.

    When ``background`` is provided, it is also scaled relative to the target
    using ``background_snr_db`` and added into the mixture, but the original
    return signature is preserved for backward compatibility.
    """
    target = ensure_float32_mono(target)
    interference = ensure_float32_mono(interference)

    if target.shape[0] != interference.shape[0]:
        raise ValueError(
            "Target and interference must have the same length before mixing. "
            f"Got {target.shape[0]} and {interference.shape[0]}."
        )

    background_scaled: Optional[np.ndarray] = None
    if background is not None:
        background = ensure_float32_mono(background)
        if background.shape[0] != target.shape[0]:
            raise ValueError(
                "Background must have the same length as the target before mixing. "
                f"Got {background.shape[0]} and {target.shape[0]}."
            )

    if zero_mean_before_mix:
        target = zero_mean(target)
        interference = zero_mean(interference)
        if background is not None:
            background = zero_mean(background)

    interference_scaled = scale_interference_to_snr(target, interference, snr_db, eps=eps)
    mixture = target + interference_scaled

    if background is not None:
        if background_snr_db is None:
            raise ValueError("background_snr_db must be provided when background is used.")
        background_scaled = scale_signal_to_reference_snr(target, background, background_snr_db, eps=eps)
        mixture = mixture + background_scaled

    if peak_norm:
        mixture, target, interference_scaled, _, _ = _joint_peak_normalize(
            mixture=mixture,
            target=target,
            interference=interference_scaled,
            background=background_scaled,
            peak_target=peak_target,
            eps=eps,
        )

    return (
        np.ascontiguousarray(mixture.astype(np.float32, copy=False)),
        np.ascontiguousarray(target.astype(np.float32, copy=False)),
        np.ascontiguousarray(interference_scaled.astype(np.float32, copy=False)),
    )


def _resolve_config(
    config: Optional[MixingConfig],
    *,
    sample_rate: int = 16000,
    segment_seconds: float = 2.0,
    snr_min_db: float = -5.0,
    snr_max_db: float = 5.0,
    peak_norm: bool = True,
    zero_mean_before_mix: bool = True,
) -> MixingConfig:
    """Create a config from explicit args when one is not provided."""
    if config is not None:
        return config
    return MixingConfig(
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        snr_min_db=snr_min_db,
        snr_max_db=snr_max_db,
        peak_normalize=peak_norm,
        zero_mean=zero_mean_before_mix,
    )


def build_training_mixture(
    target_wave: ArrayLike,
    interference_wave: ArrayLike,
    config: Optional[MixingConfig] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    background_wave: Optional[ArrayLike] = None,
    shared_rir_wave: Optional[ArrayLike] = None,
    target_rir_wave: Optional[ArrayLike] = None,
    interference_rir_wave: Optional[ArrayLike] = None,
    background_rir_wave: Optional[ArrayLike] = None,
) -> Dict[str, Union[np.ndarray, float, int, bool, None]]:
    """Build one synthetic training sample for source-separation proxy learning."""
    config = config or MixingConfig()
    rng = rng or np.random.default_rng()
    seg_len = config.segment_samples

    target_seg = random_segment(target_wave, seg_len, rng=rng, pad_if_short=True)
    interference_seg = random_segment(interference_wave, seg_len, rng=rng, pad_if_short=True)

    use_shared_rir = shared_rir_wave is not None
    target_rir = shared_rir_wave if use_shared_rir else target_rir_wave
    interference_rir = shared_rir_wave if use_shared_rir else interference_rir_wave
    background_rir = shared_rir_wave if use_shared_rir else background_rir_wave

    target_seg, target_rir_applied = maybe_apply_rir(
        target_seg,
        target_rir,
        enabled=config.apply_rir_to_target,
        probability=config.rir_probability,
        normalize_kernel=config.normalize_rir,
        rng=rng,
    )
    interference_seg, interference_rir_applied = maybe_apply_rir(
        interference_seg,
        interference_rir,
        enabled=config.apply_rir_to_interference,
        probability=config.rir_probability,
        normalize_kernel=config.normalize_rir,
        rng=rng,
    )

    snr_db = sample_snr_db(config.snr_min_db, config.snr_max_db, rng=rng)

    background_seg: Optional[np.ndarray] = None
    background_snr_db: Optional[float] = None
    background_used = False
    background_rir_applied = False

    if background_wave is not None and should_apply(config.background_probability, rng=rng):
        background_seg = random_segment(background_wave, seg_len, rng=rng, pad_if_short=True)
        background_seg, background_rir_applied = maybe_apply_rir(
            background_seg,
            background_rir,
            enabled=config.apply_rir_to_background,
            probability=config.rir_probability,
            normalize_kernel=config.normalize_rir,
            rng=rng,
        )
        if config.background_snr_min_db is None or config.background_snr_max_db is None:
            raise ValueError(
                "background_wave was provided, but background_snr_min_db/max_db are not set in MixingConfig."
            )
        background_snr_db = sample_snr_db(
            config.background_snr_min_db,
            config.background_snr_max_db,
            rng=rng,
        )
        background_used = True

    mix_wave, target_seg, interference_seg = mix_signals(
        target=target_seg,
        interference=interference_seg,
        snr_db=snr_db,
        peak_norm=config.peak_normalize,
        zero_mean_before_mix=config.zero_mean,
        background=background_seg,
        background_snr_db=background_snr_db,
        peak_target=config.peak_target,
    )

    return {
        "mix_wave": mix_wave,
        "target_wave": target_seg,
        "interference_wave": interference_seg,
        "snr_db": snr_db,
        "segment_length": seg_len,
        "background_used": background_used,
        "background_snr_db": background_snr_db,
        "shared_rir_provided": use_shared_rir,
        "target_rir_applied": target_rir_applied,
        "interference_rir_applied": interference_rir_applied,
        "background_rir_applied": background_rir_applied,
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
    config: Optional[MixingConfig] = None,
    *,
    background_wave: Optional[ArrayLike] = None,
    background_snr_db: Optional[float] = None,
    shared_rir_wave: Optional[ArrayLike] = None,
    target_rir_wave: Optional[ArrayLike] = None,
    interference_rir_wave: Optional[ArrayLike] = None,
    background_rir_wave: Optional[ArrayLike] = None,
) -> Dict[str, Union[np.ndarray, float, int, bool, None]]:
    """Build a mixture at a specified SNR."""
    rng = rng or np.random.default_rng()
    config = _resolve_config(
        config,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        snr_min_db=float(snr_db),
        snr_max_db=float(snr_db),
        peak_norm=peak_norm,
        zero_mean_before_mix=zero_mean_before_mix,
    )
    seg_len = config.segment_samples

    target_seg = random_segment(target_wave, seg_len, rng=rng, pad_if_short=True)
    interference_seg = random_segment(interference_wave, seg_len, rng=rng, pad_if_short=True)

    use_shared_rir = shared_rir_wave is not None
    target_rir = shared_rir_wave if use_shared_rir else target_rir_wave
    interference_rir = shared_rir_wave if use_shared_rir else interference_rir_wave
    background_rir = shared_rir_wave if use_shared_rir else background_rir_wave

    target_seg, target_rir_applied = maybe_apply_rir(
        target_seg,
        target_rir,
        enabled=config.apply_rir_to_target,
        probability=config.rir_probability,
        normalize_kernel=config.normalize_rir,
        rng=rng,
    )
    interference_seg, interference_rir_applied = maybe_apply_rir(
        interference_seg,
        interference_rir,
        enabled=config.apply_rir_to_interference,
        probability=config.rir_probability,
        normalize_kernel=config.normalize_rir,
        rng=rng,
    )

    background_seg: Optional[np.ndarray] = None
    background_used = False
    background_rir_applied = False
    if background_wave is not None:
        if background_snr_db is None:
            raise ValueError("background_snr_db must be provided when background_wave is used.")
        background_seg = random_segment(background_wave, seg_len, rng=rng, pad_if_short=True)
        background_seg, background_rir_applied = maybe_apply_rir(
            background_seg,
            background_rir,
            enabled=config.apply_rir_to_background,
            probability=config.rir_probability,
            normalize_kernel=config.normalize_rir,
            rng=rng,
        )
        background_used = True

    mix_wave, target_seg, interference_seg = mix_signals(
        target=target_seg,
        interference=interference_seg,
        snr_db=float(snr_db),
        peak_norm=config.peak_normalize,
        zero_mean_before_mix=config.zero_mean,
        background=background_seg,
        background_snr_db=background_snr_db,
        peak_target=config.peak_target,
    )

    return {
        "mix_wave": mix_wave,
        "target_wave": target_seg,
        "interference_wave": interference_seg,
        "snr_db": float(snr_db),
        "segment_length": seg_len,
        "background_used": background_used,
        "background_snr_db": background_snr_db,
        "shared_rir_provided": use_shared_rir,
        "target_rir_applied": target_rir_applied,
        "interference_rir_applied": interference_rir_applied,
        "background_rir_applied": background_rir_applied,
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
    "EPS",
    "MixingConfig",
    "build_fixed_snr_mixture",
    "build_training_mixture",
    "convolve_with_rir",
    "ensure_float32_mono",
    "estimate_realized_snr_db",
    "match_length",
    "maybe_apply_rir",
    "mix_signals",
    "normalize_rir_kernel",
    "peak_normalize",
    "random_segment",
    "rms",
    "sample_snr_db",
    "scale_interference_to_snr",
    "scale_signal_to_reference_snr",
    "should_apply",
    "zero_mean",
]
