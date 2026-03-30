"""Frontend utilities for source-separation proxy training.

This module provides a small, project-friendly STFT frontend for the ASD
source-separation pipeline. It is intentionally lightweight and only depends on
PyTorch.

Design goals
------------
1. Accept raw waveform batches from ``SeparationDataset`` or ``RawWaveDataset``.
2. Convert waveforms to separator inputs such as magnitude or log-magnitude
   spectrograms with shape ``[B, 1, F, T]``.
3. Keep enough metadata (mixture complex STFT / phase) to reconstruct a target
   waveform from the separator output.
4. Expose helper methods that are easy to use inside a LightningModule.

Recommended first usage
-----------------------
- separator input: mixture magnitude spectrogram
- separator target: target magnitude spectrogram
- reconstruction: predicted magnitude + mixture phase
- train loss: L1(mag) + SI-SDR(wave)

This is a pragmatic first implementation for the MVP. It is not tied to a
specific backbone, so the same frontend can be reused with ResUNet, DPRNN-
augmented models, or TF-GridNet-style experiments later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

EPS = 1e-8


@dataclass(frozen=True)
class FrontendConfig:
    """Configuration for :class:`STFTFrontend`.

    Attributes
    ----------
    sample_rate:
        Audio sample rate.
    n_fft:
        FFT size.
    hop_length:
        Hop length used in STFT/iSTFT.
    win_length:
        Window length. If ``None``, defaults to ``n_fft``.
    window:
        Currently supports ``"hann"`` only.
    center:
        Forwarded to ``torch.stft`` / ``torch.istft``.
    normalized:
        Whether to use normalized STFT.
    input_representation:
        Separator input representation.
        - ``"magnitude"``: linear magnitude
        - ``"log_magnitude"``: ``log1p(magnitude)``
        - ``"power"``: ``magnitude ** 2``
    target_representation:
        Target spectrogram representation returned by ``wave_to_target_spec``.
        Usually keep this identical to ``input_representation`` for a simple
        magnitude-mask separator.
    mag_eps:
        Numerical epsilon used in magnitude-related operations.
    """

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = "hann"
    center: bool = True
    normalized: bool = False
    input_representation: str = "magnitude"
    target_representation: str = "magnitude"
    mag_eps: float = EPS


class STFTFrontend(nn.Module):
    """STFT frontend for separator training and feature extraction.

    Expected waveform shapes
    ------------------------
    - ``[B, T]``
    - ``[B, 1, T]``
    - ``[T]``

    Main workflow
    -------------
    1. ``mix_wave -> mix_input_spec`` using :meth:`wave_to_input_spec`
    2. separator predicts ``pred_target_spec``
    3. ``pred_target_spec + mix phase -> pred_target_wave`` using
       :meth:`pred_spec_to_wave`

    The separator output is assumed to be a single-channel spectrogram with
    shape ``[B, 1, F, T]``. This fits the intended MVP where the model predicts
    a target magnitude-like spectrogram.
    """

    SUPPORTED_REPRESENTATIONS = {"magnitude", "log_magnitude", "power"}

    def __init__(self, config: Optional[FrontendConfig] = None, **kwargs) -> None:
        super().__init__()

        if config is None:
            config = FrontendConfig(**kwargs)
        self.config = config

        if self.config.input_representation not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported input_representation={self.config.input_representation!r}. "
                f"Supported: {sorted(self.SUPPORTED_REPRESENTATIONS)}"
            )
        if self.config.target_representation not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported target_representation={self.config.target_representation!r}. "
                f"Supported: {sorted(self.SUPPORTED_REPRESENTATIONS)}"
            )

        win_length = self.config.n_fft if self.config.win_length is None else self.config.win_length
        if self.config.window != "hann":
            raise ValueError("Only 'hann' window is supported in this MVP frontend.")

        window = torch.hann_window(win_length)
        self.register_buffer("window", window, persistent=False)
        self.win_length = win_length

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_wave_batch(wave: torch.Tensor) -> torch.Tensor:
        """Convert waveform to shape ``[B, T]``.

        Accepted inputs
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
        raise ValueError(
            f"Expected waveform shape [T], [B,T], or [B,1,T], got {tuple(wave.shape)}."
        )

    @staticmethod
    def _ensure_spec_batch(spec: torch.Tensor) -> torch.Tensor:
        """Convert separator spectrogram output to shape ``[B, 1, F, T]``."""
        if not torch.is_tensor(spec):
            raise TypeError(f"Expected torch.Tensor, got {type(spec)!r}.")

        if spec.ndim == 3:
            return spec.unsqueeze(1)
        if spec.ndim == 4 and spec.shape[1] == 1:
            return spec
        raise ValueError(
            f"Expected separator spectrogram shape [B,F,T] or [B,1,F,T], got {tuple(spec.shape)}."
        )

    # ------------------------------------------------------------------
    # Core STFT / iSTFT
    # ------------------------------------------------------------------
    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        """Compute complex STFT with output shape ``[B, F, T]``."""
        wave = self._ensure_wave_batch(wave)
        return torch.stft(
            wave,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wave.device, dtype=wave.dtype),
            center=self.config.center,
            normalized=self.config.normalized,
            onesided=True,
            return_complex=True,
        )

    def istft(self, complex_spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Invert a complex STFT tensor of shape ``[B, F, T]`` to ``[B, T]``."""
        if not torch.is_tensor(complex_spec):
            raise TypeError(f"Expected torch.Tensor, got {type(complex_spec)!r}.")
        if complex_spec.ndim != 3:
            raise ValueError(
                f"Expected complex spectrogram shape [B,F,T], got {tuple(complex_spec.shape)}."
            )

        return torch.istft(
            complex_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=complex_spec.device, dtype=complex_spec.real.dtype),
            center=self.config.center,
            normalized=self.config.normalized,
            onesided=True,
            length=length,
        )

    # ------------------------------------------------------------------
    # Representation transforms
    # ------------------------------------------------------------------
    def complex_to_magnitude(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Return linear magnitude with shape ``[B, F, T]``."""
        return complex_spec.abs().clamp_min(self.config.mag_eps)

    def complex_to_phase(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Return unit-magnitude phase tensor with shape ``[B, F, T]``."""
        mag = complex_spec.abs().clamp_min(self.config.mag_eps)
        return complex_spec / mag

    def apply_representation(self, magnitude: torch.Tensor, representation: str) -> torch.Tensor:
        """Convert linear magnitude to the requested representation."""
        if representation == "magnitude":
            return magnitude
        if representation == "log_magnitude":
            return torch.log1p(magnitude)
        if representation == "power":
            return magnitude.pow(2.0)
        raise ValueError(f"Unsupported representation={representation!r}.")

    def invert_representation(self, spec: torch.Tensor, representation: str) -> torch.Tensor:
        """Convert a represented spectrogram back to linear magnitude."""
        if representation == "magnitude":
            mag = spec
        elif representation == "log_magnitude":
            mag = torch.expm1(spec)
        elif representation == "power":
            mag = torch.sqrt(spec.clamp_min(0.0))
        else:
            raise ValueError(f"Unsupported representation={representation!r}.")

        return mag.clamp_min(self.config.mag_eps)

    # ------------------------------------------------------------------
    # Public helpers used by the separator pipeline
    # ------------------------------------------------------------------
    def wave_to_input_spec(self, wave: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert waveform to separator input spec.

        Returns
        -------
        input_spec:
            Tensor with shape ``[B, 1, F, T]``.
        aux:
            Dictionary containing reusable STFT metadata:
            - ``complex_spec``: complex STFT ``[B,F,T]``
            - ``phase``: unit phase ``[B,F,T]``
            - ``magnitude``: linear magnitude ``[B,F,T]``
            - ``length``: original waveform length ``[B]``
        """
        wave_b = self._ensure_wave_batch(wave)
        complex_spec = self.stft(wave_b)
        magnitude = self.complex_to_magnitude(complex_spec)
        phase = self.complex_to_phase(complex_spec)
        input_spec = self.apply_representation(magnitude, self.config.input_representation).unsqueeze(1)

        aux = {
            "complex_spec": complex_spec,
            "phase": phase,
            "magnitude": magnitude,
            "length": torch.full(
                (wave_b.shape[0],), wave_b.shape[-1], device=wave_b.device, dtype=torch.long
            ),
        }
        return input_spec, aux

    def wave_to_target_spec(self, wave: torch.Tensor) -> torch.Tensor:
        """Convert target waveform to a separator target spec ``[B,1,F,T]``."""
        complex_spec = self.stft(wave)
        magnitude = self.complex_to_magnitude(complex_spec)
        return self.apply_representation(magnitude, self.config.target_representation).unsqueeze(1)

    def pred_spec_to_magnitude(self, pred_spec: torch.Tensor) -> torch.Tensor:
        """Convert model output back to linear magnitude.

        Notes
        -----
        The separator output is interpreted according to
        ``config.target_representation``. This means your model should predict
        the same representation that ``wave_to_target_spec`` returns.
        """
        pred_spec = self._ensure_spec_batch(pred_spec)
        pred_mag = self.invert_representation(pred_spec[:, 0], self.config.target_representation)
        return pred_mag

    def pred_spec_to_wave(
        self,
        pred_spec: torch.Tensor,
        reference_aux: Dict[str, torch.Tensor],
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """Reconstruct waveform from predicted target spec using reference phase.

        Parameters
        ----------
        pred_spec:
            Separator output with shape ``[B,1,F,T]`` or ``[B,F,T]``.
        reference_aux:
            Auxiliary dictionary returned by :meth:`wave_to_input_spec`. The
            reconstruction uses ``reference_aux['phase']``.
        length:
            Optional waveform length. If omitted, uses the common batch length
            stored in ``reference_aux['length']``.
        """
        pred_mag = self.pred_spec_to_magnitude(pred_spec)
        phase = reference_aux["phase"]
        complex_pred = pred_mag * phase

        if length is None:
            if "length" in reference_aux:
                # In the current pipeline each batch uses the same fixed segment
                # length, so it is safe to use the first element.
                length = int(reference_aux["length"][0].item())
        return self.istft(complex_pred, length=length)

    def make_target_mask(
        self,
        mix_wave: torch.Tensor,
        target_wave: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Create an oracle target mask for debugging or auxiliary supervision.

        Returns a mask with shape ``[B,1,F,T]`` defined as
            target_mag / mix_mag
        with optional clipping to ``[0, 1]``.
        """
        mix_complex = self.stft(mix_wave)
        target_complex = self.stft(target_wave)
        mix_mag = self.complex_to_magnitude(mix_complex)
        target_mag = self.complex_to_magnitude(target_complex)
        mask = target_mag / mix_mag.clamp_min(self.config.mag_eps)
        if clamp:
            mask = mask.clamp(0.0, 1.0)
        return mask.unsqueeze(1)

    def forward(self, wave: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Alias of :meth:`wave_to_input_spec` for convenience."""
        return self.wave_to_input_spec(wave)


def build_frontend(**kwargs) -> STFTFrontend:
    """Small helper to instantiate :class:`STFTFrontend` from keyword args."""
    return STFTFrontend(FrontendConfig(**kwargs))
