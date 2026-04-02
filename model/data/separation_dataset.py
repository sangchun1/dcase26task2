"""Dataset utilities for source-separation proxy training.

This dataset is designed for the ASD project path where we replace the old
classification proxy with a source-separation proxy.

Main behavior
-------------
- reads a CSV such as dev_train.csv / pretrain_6.csv
- treats each row as a target normal machine clip
- samples an interference clip from a configurable pool
- builds an on-the-fly synthetic mixture using model.utils.mixing
- optionally returns guide-conditioning metadata for stage-2 SED / Time-FiLM
- optionally returns a reference waveform for future query-conditioned variants
- returns Torch tensors and useful metadata

Typical use
-----------
>>> ds = SeparationDataset(
...     csv_path="data/dev_train.csv",
...     sample_rate=16000,
...     segment_seconds=2.0,
...     interference_mode="other_machine",
...     guide_class_mode="machine",
... )
>>> batch = ds[0]
>>> batch["mix_wave"].shape
 torch.Size([1, 32000])

The default recommendation for the first experiment is:
- target: normal train clip from the current row
- interference: normal clip from a different machine
- guide class: machine identity
- random SNR: Uniform(-5, 5)
- segment length: 2.0 s
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

SUPPORTED_GUIDE_CLASS_MODES = {
    "single",
    "machine",
    "domain",
    "attribute",
    "year",
    "machine_domain",
    "machine_attribute",
    "machine_domain_attribute",
    "machine_year",
    "custom_column",
}

SUPPORTED_REFERENCE_MODES = {
    "same_target_class",
    "same_machine",
    "same_domain",
    "same_attribute",
    "self",
    "disabled",
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
    guide_class_name: str
    guide_class_index: int
    row_index: int


def _is_stackable_tensor(value: object) -> bool:
    return torch.is_tensor(value)


def separation_collate_fn(batch: Sequence[Dict]) -> Dict:
    """Collate function for :class:`SeparationDataset`.

    Tensor fields are stacked when shapes are compatible. Metadata fields stay
    as Python lists. This allows optional reference tensors to be returned
    without breaking backward compatibility.
    """

    if len(batch) == 0:
        raise ValueError("Received an empty batch in separation_collate_fn.")

    out: Dict[str, object] = {}
    first = batch[0]

    for key in first.keys():
        values = [item[key] for item in batch]
        if all(_is_stackable_tensor(v) for v in values):
            try:
                out[key] = torch.stack(values, dim=0)
                continue
            except RuntimeError:
                pass
        out[key] = values

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
    guide_class_mode:
        How to assign a target class index for guide conditioning.
        Common choice for ASD is ``"machine"``.
    guide_class_column:
        Required when ``guide_class_mode="custom_column"``.
    return_reference_wave:
        If True, sample and return a reference waveform for future
        query-conditioned experiments.
    reference_mode:
        Rule used to sample the optional reference waveform.
    reference_exclude_self:
        Whether reference sampling should avoid returning the current row.
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
        guide_class_mode: str = "machine",
        guide_class_column: Optional[str] = None,
        return_reference_wave: bool = False,
        reference_mode: str = "same_target_class",
        reference_exclude_self: bool = True,
    ) -> None:
        super().__init__()

        if interference_mode not in SUPPORTED_INTERFERENCE_MODES:
            raise ValueError(
                f"Unsupported interference_mode='{interference_mode}'. "
                f"Choose from {sorted(SUPPORTED_INTERFERENCE_MODES)}."
            )

        if guide_class_mode not in SUPPORTED_GUIDE_CLASS_MODES:
            raise ValueError(
                f"Unsupported guide_class_mode='{guide_class_mode}'. "
                f"Choose from {sorted(SUPPORTED_GUIDE_CLASS_MODES)}."
            )

        if reference_mode not in SUPPORTED_REFERENCE_MODES:
            raise ValueError(
                f"Unsupported reference_mode='{reference_mode}'. "
                f"Choose from {sorted(SUPPORTED_REFERENCE_MODES)}."
            )

        self.csv_path = str(csv_path)
        self.sample_rate = int(sample_rate)
        self.interference_mode = interference_mode
        self.fixed_snr_db = fixed_snr_db
        self.return_realized_snr = return_realized_snr
        self.guide_class_mode = str(guide_class_mode)
        self.guide_class_column = guide_class_column
        self.return_reference_wave = bool(return_reference_wave)
        self.reference_mode = str(reference_mode)
        self.reference_exclude_self = bool(reference_exclude_self)
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

        self.df = self._read_main_csv(
            self.csv_path,
            filter_normal_only=filter_normal_only,
            guide_class_mode=self.guide_class_mode,
            guide_class_column=self.guide_class_column,
        )
        if len(self.df) == 0:
            raise ValueError(f"No usable rows found in csv_path={self.csv_path}")

        self.guide_class_names, self.guide_class_to_index = self._build_guide_class_mapping(self.df)
        self.df["guide_class_name"] = self.df["guide_class_name"].astype(str)
        self.df["guide_class_index"] = self.df["guide_class_name"].map(self.guide_class_to_index).astype(int)

        self.class_to_index = self._build_class_to_index(self.df)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.num_classes = len(self.class_to_index)
        
        self.target_rows: List[SeparationRow] = [self._row_to_obj(row) for _, row in self.df.iterrows()]
        self.rows_by_machine: Dict[str, List[int]] = self._build_rows_by_machine(self.df)
        self.rows_by_guide_class: Dict[str, List[int]] = self._build_rows_by_column(self.df, "guide_class_name")
        self.rows_by_domain: Dict[str, List[int]] = self._build_rows_by_optional_column(self.df, "domain")
        self.rows_by_attribute: Dict[str, List[int]] = self._build_rows_by_optional_column(self.df, "attribute")

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
            "target_row_index": int(target_row.row_index),
            "interference_mode": self.interference_mode,
            "guide_class_mode": self.guide_class_mode,
            "guide_num_classes": int(len(self.guide_class_names)),
            "target_class_name": target_row.guide_class_name,
            "target_class_index": int(target_row.guide_class_index),
            "class_index": int(target_row.guide_class_index),
            "guide_class_index": int(target_row.guide_class_index),
        }

        if self.return_realized_snr:
            item["realized_snr_db"] = float(
                estimate_realized_snr_db(mixed["target_wave"], mixed["interference_wave"])
            )

        if self.return_reference_wave:
            reference_row = self._sample_reference_row(index=index, target_row=target_row)
            reference_wave = self._load_audio(reference_row.audio_path)
            reference_wave = self._prepare_reference_wave(reference_wave)
            item["reference_wave"] = torch.from_numpy(reference_wave).float().unsqueeze(0)
            item["reference_audio_path"] = reference_row.audio_path
            item["reference_machine"] = reference_row.machine
            item["reference_class_name"] = reference_row.guide_class_name
            item["reference_class_index"] = int(reference_row.guide_class_index)
            item["reference_row_index"] = int(reference_row.row_index)
        
        target_class_name = self._make_class_name(target_row)
        target_class_index = int(self.class_to_index[target_class_name])

        item.update({
            "target_class_name": target_class_name,
            "target_class_index": target_class_index,
            "guide_class_index": target_class_index,
            "class_index": target_class_index,
            "guide_num_classes": int(self.num_classes),
        })

        return item
    
    def _make_class_name(self, row: SeparationRow) -> str:
        mode = self.guide_class_mode
        if mode == "machine":
            return str(row.machine)
        if mode == "machine_domain":
            return f"{row.machine}::{row.domain}"
        if mode == "machine_attribute":
            return f"{row.machine}::{row.attribute}"
        return str(row.machine)

    def _build_class_to_index(self, df: pd.DataFrame) -> Dict[str, int]:
        names = sorted({
            self._make_class_name(self._row_to_obj(row))
            for _, row in df.iterrows()
        })
        return {name: idx for idx, name in enumerate(names)}

    @staticmethod
    def _read_main_csv(
        csv_path: str,
        filter_normal_only: bool,
        guide_class_mode: str,
        guide_class_column: Optional[str],
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        required = {"audio_path", "machine"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for col in ["year", "domain", "attribute", "label"]:
            if col not in df.columns:
                df[col] = None

        if guide_class_mode == "custom_column":
            if not guide_class_column:
                raise ValueError("guide_class_column must be provided when guide_class_mode='custom_column'.")
            if guide_class_column not in df.columns:
                raise ValueError(
                    f"guide_class_column='{guide_class_column}' is not present in the CSV columns."
                )
        else:
            guide_class_column = None

        if filter_normal_only and "label" in df.columns:
            labels = df["label"].astype(str).str.lower()
            df = df[labels != "anomaly"].reset_index(drop=True)

        df["audio_path"] = df["audio_path"].astype(str)
        df["machine"] = df["machine"].astype(str)
        df["guide_class_name"] = SeparationDataset._compute_guide_class_names(
            df=df,
            guide_class_mode=guide_class_mode,
            guide_class_column=guide_class_column,
        )
        return df

    @staticmethod
    def _safe_series_str(series: pd.Series, fill_value: str) -> pd.Series:
        out = series.copy()
        mask = out.isna()
        out = out.astype(str)
        if mask.any():
            out.loc[mask] = fill_value
        return out

    @staticmethod
    def _compute_guide_class_names(
        df: pd.DataFrame,
        guide_class_mode: str,
        guide_class_column: Optional[str],
    ) -> pd.Series:
        if guide_class_mode == "single":
            return pd.Series(["default"] * len(df), index=df.index)

        if guide_class_mode == "machine":
            return df["machine"].astype(str)

        if guide_class_mode == "domain":
            return SeparationDataset._safe_series_str(df["domain"], "unknown_domain")

        if guide_class_mode == "attribute":
            return SeparationDataset._safe_series_str(df["attribute"], "unknown_attribute")

        if guide_class_mode == "year":
            return SeparationDataset._safe_series_str(df["year"], "unknown_year")

        if guide_class_mode == "machine_domain":
            machine = df["machine"].astype(str)
            domain = SeparationDataset._safe_series_str(df["domain"], "unknown_domain")
            return machine + "::" + domain

        if guide_class_mode == "machine_attribute":
            machine = df["machine"].astype(str)
            attribute = SeparationDataset._safe_series_str(df["attribute"], "unknown_attribute")
            return machine + "::" + attribute

        if guide_class_mode == "machine_domain_attribute":
            machine = df["machine"].astype(str)
            domain = SeparationDataset._safe_series_str(df["domain"], "unknown_domain")
            attribute = SeparationDataset._safe_series_str(df["attribute"], "unknown_attribute")
            return machine + "::" + domain + "::" + attribute

        if guide_class_mode == "machine_year":
            machine = df["machine"].astype(str)
            year = SeparationDataset._safe_series_str(df["year"], "unknown_year")
            return machine + "::" + year

        if guide_class_mode == "custom_column":
            assert guide_class_column is not None
            return SeparationDataset._safe_series_str(df[guide_class_column], "unknown_custom")

        raise ValueError(f"Unsupported guide_class_mode='{guide_class_mode}'")

    @staticmethod
    def _build_guide_class_mapping(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
        names = sorted(df["guide_class_name"].astype(str).unique().tolist())
        mapping = {name: idx for idx, name in enumerate(names)}
        return names, mapping

    @staticmethod
    def _build_rows_by_machine(df: pd.DataFrame) -> Dict[str, List[int]]:
        rows_by_machine: Dict[str, List[int]] = {}
        for idx, row in df.iterrows():
            rows_by_machine.setdefault(str(row["machine"]), []).append(int(idx))
        return rows_by_machine

    @staticmethod
    def _build_rows_by_column(df: pd.DataFrame, column: str) -> Dict[str, List[int]]:
        rows_by_value: Dict[str, List[int]] = {}
        for idx, row in df.iterrows():
            rows_by_value.setdefault(str(row[column]), []).append(int(idx))
        return rows_by_value

    @staticmethod
    def _build_rows_by_optional_column(df: pd.DataFrame, column: str) -> Dict[str, List[int]]:
        rows_by_value: Dict[str, List[int]] = {}
        for idx, row in df.iterrows():
            value = row[column]
            if pd.isna(value):
                continue
            rows_by_value.setdefault(str(value), []).append(int(idx))
        return rows_by_value

    @staticmethod
    def _row_to_obj(row: pd.Series) -> SeparationRow:
        return SeparationRow(
            audio_path=str(row["audio_path"]),
            machine=str(row["machine"]),
            year=None if pd.isna(row.get("year")) else int(row.get("year")),
            domain=None if pd.isna(row.get("domain")) else str(row.get("domain")),
            attribute=None if pd.isna(row.get("attribute")) else str(row.get("attribute")),
            label=None if pd.isna(row.get("label")) else str(row.get("label")),
            guide_class_name=str(row.get("guide_class_name")),
            guide_class_index=int(row.get("guide_class_index")),
            row_index=int(row.name) if row.name is not None else -1,
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

            for idx, row in noise_df.iterrows():
                rows.append(
                    SeparationRow(
                        audio_path=str(row["audio_path"]),
                        machine=str(row["machine"]),
                        year=None if pd.isna(row.get("year")) else int(row.get("year")),
                        domain=None if pd.isna(row.get("domain")) else str(row.get("domain")),
                        attribute=None if pd.isna(row.get("attribute")) else str(row.get("attribute")),
                        label=None if pd.isna(row.get("label")) else str(row.get("label")),
                        guide_class_name="external_noise",
                        guide_class_index=-1,
                        row_index=int(idx),
                    )
                )

        if noise_paths is not None:
            for idx, path in enumerate(noise_paths):
                rows.append(
                    SeparationRow(
                        audio_path=str(path),
                        machine="external_noise",
                        year=None,
                        domain=None,
                        attribute=Path(path).stem,
                        label="external_noise",
                        guide_class_name="external_noise",
                        guide_class_index=-1,
                        row_index=int(idx),
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
        candidates = [row for row in self.target_rows if row.machine != target_machine]
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

    def _sample_reference_row(self, index: int, target_row: SeparationRow) -> SeparationRow:
        if self.reference_mode == "disabled":
            raise RuntimeError("Reference sampling requested while reference_mode='disabled'.")

        if self.reference_mode == "self":
            return target_row

        if self.reference_mode == "same_target_class":
            candidate_indices = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
        elif self.reference_mode == "same_machine":
            candidate_indices = list(self.rows_by_machine.get(target_row.machine, []))
        elif self.reference_mode == "same_domain":
            if target_row.domain is None:
                candidate_indices = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
            else:
                candidate_indices = list(self.rows_by_domain.get(str(target_row.domain), []))
        elif self.reference_mode == "same_attribute":
            if target_row.attribute is None:
                candidate_indices = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
            else:
                candidate_indices = list(self.rows_by_attribute.get(str(target_row.attribute), []))
        else:
            raise ValueError(f"Unsupported reference_mode='{self.reference_mode}'")

        if self.reference_exclude_self:
            candidate_indices = [i for i in candidate_indices if i != index]

        if not candidate_indices and self.reference_mode != "same_target_class":
            fallback = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
            if self.reference_exclude_self:
                fallback = [i for i in fallback if i != index]
            candidate_indices = fallback

        if not candidate_indices:
            return target_row

        chosen_index = self.py_rng.choice(candidate_indices)
        return self.target_rows[chosen_index]

    def _prepare_reference_wave(self, wave: np.ndarray) -> np.ndarray:
        segment_length = int(round(self.mix_config.segment_seconds * self.sample_rate))
        if segment_length <= 0:
            raise ValueError("segment_seconds must produce a positive number of samples.")

        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if self.mix_config.zero_mean and wave.size > 0:
            wave = wave - np.mean(wave, dtype=np.float32)

        if wave.shape[0] < segment_length:
            pad_width = segment_length - wave.shape[0]
            wave = np.pad(wave, (0, pad_width), mode="constant")
        elif wave.shape[0] > segment_length:
            max_offset = wave.shape[0] - segment_length
            start = int(self.rng.integers(0, max_offset + 1))
            wave = wave[start : start + segment_length]

        return wave.astype(np.float32, copy=False)

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
        guide_class_mode: str = "machine",
        guide_class_column: Optional[str] = None,
        return_reference_wave: bool = False,
        reference_mode: str = "same_target_class",
        reference_exclude_self: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.return_reference_wave = bool(return_reference_wave)
        self.reference_mode = str(reference_mode)
        self.reference_exclude_self = bool(reference_exclude_self)
        self.py_rng = random.Random(seed)

        self.df = SeparationDataset._read_main_csv(
            csv_path=csv_path,
            filter_normal_only=filter_normal_only,
            guide_class_mode=guide_class_mode,
            guide_class_column=guide_class_column,
        )
        self.guide_class_names, self.guide_class_to_index = SeparationDataset._build_guide_class_mapping(self.df)
        self.df["guide_class_name"] = self.df["guide_class_name"].astype(str)
        self.df["guide_class_index"] = self.df["guide_class_name"].map(self.guide_class_to_index).astype(int)
        self.rows: List[SeparationRow] = [SeparationDataset._row_to_obj(row) for _, row in self.df.iterrows()]
        self.rows_by_machine = SeparationDataset._build_rows_by_machine(self.df)
        self.rows_by_guide_class = SeparationDataset._build_rows_by_column(self.df, "guide_class_name")
        self.rows_by_domain = SeparationDataset._build_rows_by_optional_column(self.df, "domain")
        self.rows_by_attribute = SeparationDataset._build_rows_by_optional_column(self.df, "attribute")

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

        item = {
            "wave": torch.from_numpy(wave).float().unsqueeze(0),
            "audio_path": row.audio_path,
            "machine": row.machine,
            "year": row.year,
            "domain": row.domain,
            "attribute": row.attribute,
            "label": row.label,
            "target_class_name": row.guide_class_name,
            "target_class_index": int(row.guide_class_index),
            "class_index": int(row.guide_class_index),
            "guide_class_index": int(row.guide_class_index),
            "guide_num_classes": int(len(self.guide_class_names)),
            "target_row_index": int(row.row_index),
        }

        if self.return_reference_wave:
            reference_row = self._sample_reference_row(index=index, target_row=row)
            ref_wave, ref_sr = sf.read(reference_row.audio_path)
            ref_wave = ensure_float32_mono(ref_wave)
            if ref_sr != self.sample_rate:
                raise ValueError(
                    f"Sample rate mismatch for '{reference_row.audio_path}': "
                    f"expected {self.sample_rate}, got {ref_sr}."
                )
            item["reference_wave"] = torch.from_numpy(ref_wave).float().unsqueeze(0)
            item["reference_audio_path"] = reference_row.audio_path
            item["reference_machine"] = reference_row.machine
            item["reference_class_name"] = reference_row.guide_class_name
            item["reference_class_index"] = int(reference_row.guide_class_index)

        return item

    def _sample_reference_row(self, index: int, target_row: SeparationRow) -> SeparationRow:
        if self.reference_mode == "self":
            return target_row

        if self.reference_mode == "same_target_class":
            candidate_indices = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
        elif self.reference_mode == "same_machine":
            candidate_indices = list(self.rows_by_machine.get(target_row.machine, []))
        elif self.reference_mode == "same_domain":
            if target_row.domain is None:
                candidate_indices = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
            else:
                candidate_indices = list(self.rows_by_domain.get(str(target_row.domain), []))
        elif self.reference_mode == "same_attribute":
            if target_row.attribute is None:
                candidate_indices = list(self.rows_by_guide_class.get(target_row.guide_class_name, []))
            else:
                candidate_indices = list(self.rows_by_attribute.get(str(target_row.attribute), []))
        else:
            raise ValueError(f"Unsupported reference_mode='{self.reference_mode}'")

        if self.reference_exclude_self:
            candidate_indices = [i for i in candidate_indices if i != index]

        if not candidate_indices:
            return target_row
        return self.rows[self.py_rng.choice(candidate_indices)]


__all__ = [
    "SUPPORTED_INTERFERENCE_MODES",
    "SUPPORTED_GUIDE_CLASS_MODES",
    "SUPPORTED_REFERENCE_MODES",
    "SeparationDataset",
    "RawWaveDataset",
    "separation_collate_fn",
]
