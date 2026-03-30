"""Raw-wave dataset utilities for separator feature extraction and scoring.

This dataset is intentionally simple. After training the source-separation proxy,
we usually want to:

1. build a normal feature bank from development/train normal clips
2. extract embeddings from validation / test clips
3. compute anomaly scores such as Mahalanobis distance

For those steps, we do *not* need synthetic mixtures. We only need to read the
original waveform and keep the metadata attached to each file.

Typical use
-----------
>>> ds = RawWaveDataset(
...     csv_path="data/dev_test.csv",
...     sample_rate=16000,
...     filter_normal_only=False,
...     segment_seconds=None,
... )
>>> item = ds[0]
>>> item["wave"].shape
 torch.Size([1, T])

Recommended first usage
-----------------------
- normal-bank extraction:
    RawWaveDataset(csv_path="dev_train.csv", filter_normal_only=True)
- validation/test extraction:
    RawWaveDataset(csv_path="dev_test.csv", filter_normal_only=False)

Notes
-----
- This file is separate from ``separation_dataset.py`` on purpose.
  ``SeparationDataset`` is for proxy training and makes synthetic mixtures.
  ``RawWaveDataset`` is for embedding extraction on *original* audio.
- The dataset accepts optional fixed-length segmentation. If ``segment_seconds``
  is set, it returns one segment per item. If ``segment_seconds`` is None, it
  returns the full waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

try:
    from model.utils.mixing import ensure_float32_mono, match_length, random_segment
except ImportError:  # pragma: no cover
    from ..utils.mixing import ensure_float32_mono, match_length, random_segment  # type: ignore


REQUIRED_COLUMNS = {"audio_path", "machine"}
OPTIONAL_COLUMNS = ["year", "domain", "attribute", "label", "section"]


@dataclass(frozen=True)
class RawWaveRow:
    """A single CSV row parsed into a lightweight container."""

    audio_path: str
    machine: str
    year: Optional[int]
    domain: Optional[str]
    attribute: Optional[str]
    label: Optional[str]
    section: Optional[str]



def raw_wave_collate_fn(batch: Sequence[Dict]) -> Dict:
    """Collate function for :class:`RawWaveDataset`.

    Behavior
    --------
    - Tensor fields with equal temporal length are stacked.
    - Variable-length ``wave`` tensors are returned as a list instead of being
      padded silently.
    - Metadata fields stay as Python lists.

    This is safer for feature extraction because some backbones may want to
    handle variable-length audio explicitly.
    """

    if len(batch) == 0:
        raise ValueError("Received an empty batch in raw_wave_collate_fn.")

    out: Dict[str, object] = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        if key == "wave":
            lengths = [int(v.shape[-1]) for v in values]
            if len(set(lengths)) == 1:
                out[key] = torch.stack(values, dim=0)
            else:
                out[key] = values
                out["wave_lengths"] = lengths
        elif torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values

    return out


class RawWaveDataset(Dataset):
    """Dataset for reading original waveforms plus metadata from a CSV.

    Parameters
    ----------
    csv_path:
        CSV file containing at least ``audio_path`` and ``machine`` columns.
    sample_rate:
        Expected sample rate of all audio files.
    filter_normal_only:
        If True, drop rows whose ``label`` column equals ``anomaly``.
    segment_seconds:
        Optional fixed segment duration. If provided, each item returns exactly
        one segment of that duration. If omitted, the full waveform is returned.
    segment_mode:
        How to obtain a fixed segment when ``segment_seconds`` is not None.
        - ``center``: deterministic center crop / pad
        - ``random``: random crop / zero pad
    return_numpy:
        If True, return NumPy arrays instead of Torch tensors for ``wave``.
        The default is False because the rest of the current pipeline uses Torch.
    seed:
        Random seed used only when ``segment_mode='random'``.
    zero_mean:
        If True, subtract the waveform mean after loading.
    peak_normalize:
        If True, divide by the waveform peak after loading.
    """

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        filter_normal_only: bool = False,
        segment_seconds: Optional[float] = None,
        segment_mode: str = "center",
        return_numpy: bool = False,
        seed: Optional[int] = None,
        zero_mean: bool = False,
        peak_normalize: bool = False,
    ) -> None:
        super().__init__()

        if segment_mode not in {"center", "random"}:
            raise ValueError("segment_mode must be one of {'center', 'random'}.")

        self.csv_path = str(csv_path)
        self.sample_rate = int(sample_rate)
        self.filter_normal_only = bool(filter_normal_only)
        self.segment_seconds = segment_seconds
        self.segment_mode = segment_mode
        self.return_numpy = bool(return_numpy)
        self.zero_mean = bool(zero_mean)
        self.peak_normalize = bool(peak_normalize)
        self.rng = np.random.default_rng(seed)

        self.segment_samples: Optional[int]
        if segment_seconds is None:
            self.segment_samples = None
        else:
            self.segment_samples = int(round(float(segment_seconds) * self.sample_rate))
            if self.segment_samples <= 0:
                raise ValueError("segment_seconds must correspond to at least 1 sample.")

        self.df = self._read_csv(self.csv_path, filter_normal_only=self.filter_normal_only)
        if len(self.df) == 0:
            raise ValueError(f"No usable rows found in csv_path={self.csv_path}")

        self.rows: List[RawWaveRow] = [self._row_to_obj(row) for _, row in self.df.iterrows()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        wave = self._load_audio(row.audio_path)
        original_num_samples = int(wave.shape[0])

        if self.segment_samples is not None:
            if self.segment_mode == "center":
                wave = match_length(wave, self.segment_samples)
            else:
                wave = random_segment(wave, self.segment_samples, rng=self.rng)

        if self.zero_mean:
            wave = wave - np.mean(wave, dtype=np.float32)

        if self.peak_normalize:
            peak = float(np.max(np.abs(wave)))
            if peak > 1e-8:
                wave = wave / peak

        wave_out = wave if self.return_numpy else torch.from_numpy(wave).float().unsqueeze(0)

        return {
            "wave": wave_out,
            "num_samples": int(wave.shape[0]),
            "original_num_samples": original_num_samples,
            "audio_path": row.audio_path,
            "machine": row.machine,
            "year": row.year,
            "domain": row.domain,
            "attribute": row.attribute,
            "label": row.label,
            "section": row.section,
            "is_normal": row.label is None or str(row.label).lower() != "anomaly",
        }

    @staticmethod
    def _read_csv(csv_path: str, filter_normal_only: bool) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        if filter_normal_only and "label" in df.columns:
            labels = df["label"].astype(str).str.lower()
            df = df[labels != "anomaly"].reset_index(drop=True)

        df["audio_path"] = df["audio_path"].astype(str)
        df["machine"] = df["machine"].astype(str)
        return df

    @staticmethod
    def _row_to_obj(row: pd.Series) -> RawWaveRow:
        year = None
        if not pd.isna(row.get("year")):
            try:
                year = int(row.get("year"))
            except (TypeError, ValueError):
                year = None

        return RawWaveRow(
            audio_path=str(row["audio_path"]),
            machine=str(row["machine"]),
            year=year,
            domain=None if pd.isna(row.get("domain")) else str(row.get("domain")),
            attribute=None if pd.isna(row.get("attribute")) else str(row.get("attribute")),
            label=None if pd.isna(row.get("label")) else str(row.get("label")),
            section=None if pd.isna(row.get("section")) else str(row.get("section")),
        )

    def _load_audio(self, path: str) -> np.ndarray:
        wave, sr = sf.read(path)
        wave = ensure_float32_mono(wave)
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for '{path}': expected {self.sample_rate}, got {sr}."
            )
        return wave


__all__ = [
    "RawWaveRow",
    "RawWaveDataset",
    "raw_wave_collate_fn",
]
