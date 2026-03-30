"""Dataset utilities for source-separation proxy training.

This dataset is designed for the ASD project path where we replace the old
classification proxy with a source-separation proxy.

Main behavior
-------------
- reads a CSV such as dev_train.csv / pretrain_6.csv
- treats each row as a target normal machine clip
- samples an interference clip from a configurable pool
- builds an on-the-fly synthetic mixture using model.utils.mixing
- returns Torch tensors and useful metadata

Typical use
-----------
>>> ds = SeparationDataset(
...     csv_path="data/dev_train.csv",
...     sample_rate=16000,
...     segment_seconds=2.0,
...     interference_mode="other_machine",
... )
>>> batch = ds[0]
>>> batch["mix_wave"].shape
 torch.Size([1, 32000])

The default recommendation for the first experiment is:
- target: normal train clip from the current row
- interference: normal clip from a different machine
- random SNR: Uniform(-5, 5)
- segment length: 2.0 s
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

try:
    from model.utils.mixing import (
        MixingConfig,
        build_fixed_snr_mixture,
        build_training_mixture,
        ensure_float32_mono,
        estimate_realized_snr_db,
    )
except ImportError:  # pragma: no cover
    # Fallback for environments where the project root is not installed as a package.
    from ..utils.mixing import (  # type: ignore
        MixingConfig,
        build_fixed_snr_mixture,
        build_training_mixture,
        ensure_float32_mono,
        estimate_realized_snr_db,
    )


SUPPORTED_INTERFERENCE_MODES = {
    "other_machine",
    "same_machine",
    "any_machine",
    "external_noise",
    "other_machine_or_noise",
}


@dataclass(frozen=True)
class SeparationRow:
    """Lightweight row container used internally for sampling."""

    audio_path: str
    machine: str
    year: Optional[int]
    domain: Optional[str]
    attribute: Optional[str]
    label: Optional[str]



def separation_collate_fn(batch: Sequence[Dict]) -> Dict:
    """Collate function for :class:`SeparationDataset`.

    Tensor fields are stacked. Metadata fields stay as Python lists.
    """

    if len(batch) == 0:
        raise ValueError("Received an empty batch in separation_collate_fn.")

    tensor_keys = {"mix_wave", "target_wave", "interference_wave"}
    out: Dict[str, object] = {}

    for key in batch[0].keys():
        if key in tensor_keys:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            out[key] = [item[key] for item in batch]

    return out


class SeparationDataset(Dataset):
    """On-the-fly synthetic mixture dataset for source-separation proxy learning.

    Parameters
    ----------
    csv_path:
        CSV file containing at least ``audio_path`` and ``machine`` columns.
    sample_rate:
        Expected sampling rate. If a loaded waveform has a different rate, an
        error is raised.
    segment_seconds:
        Fixed segment length used for random crop / zero padding.
    snr_min_db, snr_max_db:
        Training SNR range when ``fixed_snr_db`` is not used.
    interference_mode:
        One of:
        - ``other_machine``: sample interference from a different machine
        - ``same_machine``: sample from the same machine, different row if possible
        - ``any_machine``: sample from the whole dataset
        - ``external_noise``: sample only from ``noise_csv_path`` / ``noise_paths``
        - ``other_machine_or_noise``: 50/50 mix of different machine vs external noise
    fixed_snr_db:
        If provided, the dataset always uses this SNR instead of random training
        SNR sampling. Useful for validation sets such as -5 / 0 / 5 dB.
    noise_csv_path:
        Optional CSV for external noise clips. Must contain ``audio_path`` and
        ideally ``machine``; if not present, ``machine`` is filled with
        ``"external_noise"``.
    noise_paths:
        Optional explicit list of external noise wav paths.
    filter_normal_only:
        If True, rows with label ``anomaly`` are dropped.
    seed:
        Optional seed used for reproducible interference sampling.
    return_realized_snr:
        If True, compute the realized SNR after mixing and return it.
    """

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        snr_min_db: float = -5.0,
        snr_max_db: float = 5.0,
        interference_mode: str = "other_machine",
        fixed_snr_db: Optional[float] = None,
        noise_csv_path: Optional[str] = None,
        noise_paths: Optional[Sequence[str]] = None,
        filter_normal_only: bool = True,
        seed: Optional[int] = None,
        return_realized_snr: bool = True,
    ) -> None:
        super().__init__()

        if interference_mode not in SUPPORTED_INTERFERENCE_MODES:
            raise ValueError(
                f"Unsupported interference_mode='{interference_mode}'. "
                f"Choose from {sorted(SUPPORTED_INTERFERENCE_MODES)}."
            )

        self.csv_path = str(csv_path)
        self.sample_rate = int(sample_rate)
        self.interference_mode = interference_mode
        self.fixed_snr_db = fixed_snr_db
        self.return_realized_snr = return_realized_snr
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        self.mix_config = MixingConfig(
            sample_rate=self.sample_rate,
            segment_seconds=segment_seconds,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            peak_normalize=True,
            zero_mean=True,
        )

        self.df = self._read_main_csv(self.csv_path, filter_normal_only=filter_normal_only)
        if len(self.df) == 0:
            raise ValueError(f"No usable rows found in csv_path={self.csv_path}")

        self.target_rows: List[SeparationRow] = [self._row_to_obj(row) for _, row in self.df.iterrows()]
        self.rows_by_machine: Dict[str, List[int]] = self._build_rows_by_machine(self.df)

        self.noise_rows: List[SeparationRow] = self._load_noise_rows(
            noise_csv_path=noise_csv_path,
            noise_paths=noise_paths,
        )

        if self.interference_mode in {"external_noise", "other_machine_or_noise"} and len(self.noise_rows) == 0:
            raise ValueError(
                f"interference_mode='{self.interference_mode}' requires external noise, "
                "but no noise_csv_path/noise_paths were provided."
            )

    def __len__(self) -> int:
        return len(self.target_rows)

    def __getitem__(self, index: int) -> Dict:
        target_row = self.target_rows[index]
        target_wave = self._load_audio(target_row.audio_path)

        interference_row = self._sample_interference_row(index=index, target_row=target_row)
        interference_wave = self._load_audio(interference_row.audio_path)

        if self.fixed_snr_db is None:
            mixed = build_training_mixture(
                target_wave=target_wave,
                interference_wave=interference_wave,
                config=self.mix_config,
                rng=self.rng,
            )
        else:
            mixed = build_fixed_snr_mixture(
                target_wave=target_wave,
                interference_wave=interference_wave,
                snr_db=float(self.fixed_snr_db),
                config=self.mix_config,
                rng=self.rng,
            )

        mix_wave = torch.from_numpy(mixed["mix_wave"]).float().unsqueeze(0)
        target_wave_t = torch.from_numpy(mixed["target_wave"]).float().unsqueeze(0)
        interference_wave_t = torch.from_numpy(mixed["interference_wave"]).float().unsqueeze(0)

        item = {
            "mix_wave": mix_wave,
            "target_wave": target_wave_t,
            "interference_wave": interference_wave_t,
            "snr_db": float(mixed["snr_db"]),
            "segment_length": int(mixed["segment_length"]),
            "target_audio_path": target_row.audio_path,
            "interference_audio_path": interference_row.audio_path,
            "target_machine": target_row.machine,
            "interference_machine": interference_row.machine,
            "target_year": target_row.year,
            "target_domain": target_row.domain,
            "target_attribute": target_row.attribute,
            "target_label": target_row.label,
            "interference_mode": self.interference_mode,
        }

        if self.return_realized_snr:
            item["realized_snr_db"] = float(
                estimate_realized_snr_db(mixed["target_wave"], mixed["interference_wave"])
            )

        return item

    @staticmethod
    def _read_main_csv(csv_path: str, filter_normal_only: bool) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        required = {"audio_path", "machine"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for col in ["year", "domain", "attribute", "label"]:
            if col not in df.columns:
                df[col] = None

        if filter_normal_only and "label" in df.columns:
            labels = df["label"].astype(str).str.lower()
            df = df[labels != "anomaly"].reset_index(drop=True)

        df["audio_path"] = df["audio_path"].astype(str)
        df["machine"] = df["machine"].astype(str)
        return df

    @staticmethod
    def _build_rows_by_machine(df: pd.DataFrame) -> Dict[str, List[int]]:
        rows_by_machine: Dict[str, List[int]] = {}
        for idx, row in df.iterrows():
            rows_by_machine.setdefault(str(row["machine"]), []).append(int(idx))
        return rows_by_machine

    @staticmethod
    def _row_to_obj(row: pd.Series) -> SeparationRow:
        return SeparationRow(
            audio_path=str(row["audio_path"]),
            machine=str(row["machine"]),
            year=None if pd.isna(row.get("year")) else int(row.get("year")),
            domain=None if pd.isna(row.get("domain")) else str(row.get("domain")),
            attribute=None if pd.isna(row.get("attribute")) else str(row.get("attribute")),
            label=None if pd.isna(row.get("label")) else str(row.get("label")),
        )

    def _load_noise_rows(
        self,
        noise_csv_path: Optional[str],
        noise_paths: Optional[Sequence[str]],
    ) -> List[SeparationRow]:
        rows: List[SeparationRow] = []

        if noise_csv_path is not None:
            noise_df = pd.read_csv(noise_csv_path)
            if "audio_path" not in noise_df.columns:
                raise ValueError("noise_csv_path must contain an 'audio_path' column.")

            if "machine" not in noise_df.columns:
                noise_df["machine"] = "external_noise"
            for col in ["year", "domain", "attribute", "label"]:
                if col not in noise_df.columns:
                    noise_df[col] = None

            rows.extend(self._row_to_obj(row) for _, row in noise_df.iterrows())

        if noise_paths is not None:
            for path in noise_paths:
                rows.append(
                    SeparationRow(
                        audio_path=str(path),
                        machine="external_noise",
                        year=None,
                        domain=None,
                        attribute=Path(path).stem,
                        label="external_noise",
                    )
                )

        return rows

    def _sample_interference_row(self, index: int, target_row: SeparationRow) -> SeparationRow:
        if self.interference_mode == "other_machine":
            return self._sample_other_machine_row(target_machine=target_row.machine)

        if self.interference_mode == "same_machine":
            return self._sample_same_machine_row(index=index, target_machine=target_row.machine)

        if self.interference_mode == "any_machine":
            return self._sample_any_row(exclude_index=index)

        if self.interference_mode == "external_noise":
            return self._sample_external_noise_row()

        if self.interference_mode == "other_machine_or_noise":
            if len(self.noise_rows) > 0 and self.py_rng.random() < 0.5:
                return self._sample_external_noise_row()
            return self._sample_other_machine_row(target_machine=target_row.machine)

        raise RuntimeError(f"Unexpected interference_mode='{self.interference_mode}'")

    def _sample_other_machine_row(self, target_machine: str) -> SeparationRow:
        candidates = [
            row for row in self.target_rows
            if row.machine != target_machine
        ]
        if not candidates:
            raise ValueError(
                f"No candidate rows found for interference_mode='other_machine' "
                f"with target_machine='{target_machine}'."
            )
        return self.py_rng.choice(candidates)

    def _sample_same_machine_row(self, index: int, target_machine: str) -> SeparationRow:
        candidate_indices = [i for i in self.rows_by_machine.get(target_machine, []) if i != index]
        if not candidate_indices:
            candidate_indices = self.rows_by_machine.get(target_machine, [])
        if not candidate_indices:
            raise ValueError(
                f"No candidate rows found for interference_mode='same_machine' "
                f"with target_machine='{target_machine}'."
            )
        chosen_index = self.py_rng.choice(candidate_indices)
        return self.target_rows[chosen_index]

    def _sample_any_row(self, exclude_index: Optional[int] = None) -> SeparationRow:
        if len(self.target_rows) == 0:
            raise ValueError("No rows available for interference sampling.")

        if exclude_index is None or len(self.target_rows) == 1:
            return self.py_rng.choice(self.target_rows)

        chosen_index = exclude_index
        while chosen_index == exclude_index:
            chosen_index = self.py_rng.randrange(len(self.target_rows))
        return self.target_rows[chosen_index]

    def _sample_external_noise_row(self) -> SeparationRow:
        if len(self.noise_rows) == 0:
            raise ValueError("No external noise rows available.")
        return self.py_rng.choice(self.noise_rows)

    def _load_audio(self, path: str) -> np.ndarray:
        wave, sr = sf.read(path)
        wave = ensure_float32_mono(wave)
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for '{path}': expected {self.sample_rate}, got {sr}."
            )
        return wave


class RawWaveDataset(Dataset):
    """Simple raw-wave dataset for feature bank extraction after separator training.

    This class is placed in the same file for convenience, because the next step
    in the pipeline is usually:
    1) train separator with ``SeparationDataset``
    2) extract features from raw normal/test clips using ``RawWaveDataset``
    """

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        filter_normal_only: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.df = SeparationDataset._read_main_csv(
            csv_path=csv_path,
            filter_normal_only=filter_normal_only,
        )
        self.rows: List[SeparationRow] = [SeparationDataset._row_to_obj(row) for _, row in self.df.iterrows()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        wave, sr = sf.read(row.audio_path)
        wave = ensure_float32_mono(wave)
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for '{row.audio_path}': expected {self.sample_rate}, got {sr}."
            )

        return {
            "wave": torch.from_numpy(wave).float().unsqueeze(0),
            "audio_path": row.audio_path,
            "machine": row.machine,
            "year": row.year,
            "domain": row.domain,
            "attribute": row.attribute,
            "label": row.label,
        }


__all__ = [
    "SUPPORTED_INTERFERENCE_MODES",
    "SeparationDataset",
    "RawWaveDataset",
    "separation_collate_fn",
]
