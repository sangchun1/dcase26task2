"""Feature-bank utilities for separator-based ASD embeddings.

This module is intentionally scorer-agnostic.
It sits between:

1. feature extraction (separator + feature head)
2. anomaly scoring (Mahalanobis, kNN, centroid distance, etc.)

Why this file exists
--------------------
For the new source-separation proxy pipeline, we need a clean way to:
- accumulate embeddings across batches
- keep the metadata aligned with those embeddings
- filter to normal-only subsets
- split banks by machine / domain
- save and reload banks for later scoring or analysis

Typical usage
-------------
>>> accumulator = FeatureBankAccumulator()
>>> accumulator.append(batch_embeddings, batch_metadata)
>>> bank = accumulator.finalize(name="dev_train_bank")
>>> normal_bank = bank.filter_normal_only()
>>> machine_banks = normal_bank.split_by_machine()
>>> bank.save("workspace/feature_banks/dev_train_bank")

Then later:
>>> bank = FeatureBank.load("workspace/feature_banks/dev_train_bank")
>>> print(bank.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import json

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float]]
MetadataLike = Union[pd.DataFrame, Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]]

OPTIONAL_METADATA_COLUMNS = [
    "audio_path",
    "machine",
    "domain",
    "attribute",
    "section",
    "year",
    "label",
]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------

def _to_numpy_2d(x: Union[np.ndarray, Any]) -> np.ndarray:
    """Convert torch/numpy-like embeddings to a contiguous float32 array [N, D]."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {x.shape}.")
    return np.ascontiguousarray(x)



def _resolve_metadata(
    metadata: Optional[MetadataLike] = None,
    csv_path: Optional[Union[str, Path]] = None,
    split_name: str = "metadata",
) -> pd.DataFrame:
    """Resolve metadata from a dataframe-like object or CSV path."""
    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            df = metadata.copy()
        else:
            df = pd.DataFrame(metadata)
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(
            f"{split_name} was not provided. Pass either a DataFrame-like object or {split_name}_csv_path."
        )
    return df.reset_index(drop=True)



def _normalize_batch_metadata(batch_metadata: Any, expected_len: int) -> pd.DataFrame:
    """Normalize batch metadata into a dataframe with exactly ``expected_len`` rows.

    Accepted input styles
    ---------------------
    - pandas.DataFrame
    - sequence of dicts
    - dict of lists  (typical collate_fn output)
    - dict of scalars (single-item fallback)
    """
    if isinstance(batch_metadata, pd.DataFrame):
        df = batch_metadata.copy()
    elif isinstance(batch_metadata, Mapping):
        normalized: Dict[str, List[Any]] = {}
        for key, value in batch_metadata.items():
            if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
                normalized[key] = [value] * expected_len
                continue

            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()

            value = list(value)
            if len(value) == expected_len:
                normalized[key] = value
            elif len(value) == 1 and expected_len > 1:
                normalized[key] = value * expected_len
            else:
                raise ValueError(
                    f"Metadata field '{key}' has length {len(value)} but expected {expected_len}."
                )
        df = pd.DataFrame(normalized)
    else:
        df = pd.DataFrame(batch_metadata)

    if len(df) != expected_len:
        raise ValueError(
            f"Batch metadata length mismatch: len(metadata)={len(df)} vs expected_len={expected_len}."
        )
    return df.reset_index(drop=True)



def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OPTIONAL_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out



def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return (x / norm).astype(np.float32)


# -----------------------------------------------------------------------------
# Main bank object
# -----------------------------------------------------------------------------
@dataclass
class FeatureBank:
    """Container for embeddings plus aligned metadata.

    Parameters
    ----------
    embeddings:
        Array with shape ``[N, D]``.
    metadata:
        DataFrame with ``N`` rows aligned to ``embeddings``.
    name:
        Optional descriptive name for logging / saved bank manifests.
    normalized:
        Whether embeddings are already L2-normalized row-wise.
    """

    embeddings: np.ndarray
    metadata: pd.DataFrame
    name: str = "feature_bank"
    normalized: bool = False
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.embeddings = _to_numpy_2d(self.embeddings)
        self.metadata = _ensure_optional_columns(self.metadata.reset_index(drop=True))
        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                f"FeatureBank alignment mismatch: len(embeddings)={len(self.embeddings)} vs "
                f"len(metadata)={len(self.metadata)}."
            )

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.embeddings.shape)  # type: ignore[return-value]

    @property
    def machines(self) -> List[str]:
        if "machine" not in self.metadata.columns:
            return []
        return sorted(self.metadata["machine"].dropna().astype(str).unique().tolist())

    @property
    def domains(self) -> List[str]:
        if "domain" not in self.metadata.columns:
            return []
        return sorted(self.metadata["domain"].dropna().astype(str).unique().tolist())

    def copy(self, name: Optional[str] = None) -> "FeatureBank":
        return FeatureBank(
            embeddings=self.embeddings.copy(),
            metadata=self.metadata.copy(),
            name=self.name if name is None else name,
            normalized=self.normalized,
            extra_info=dict(self.extra_info),
        )

    def with_embeddings(self, embeddings: ArrayLike, name: Optional[str] = None, normalized: Optional[bool] = None) -> "FeatureBank":
        emb = _to_numpy_2d(embeddings)
        if len(emb) != len(self.metadata):
            raise ValueError(
                f"New embeddings length mismatch: len(embeddings)={len(emb)} vs len(metadata)={len(self.metadata)}."
            )
        return FeatureBank(
            embeddings=emb,
            metadata=self.metadata.copy(),
            name=self.name if name is None else name,
            normalized=self.normalized if normalized is None else bool(normalized),
            extra_info=dict(self.extra_info),
        )

    def l2_normalize(self, inplace: bool = False) -> "FeatureBank":
        emb = _l2_normalize_rows(self.embeddings)
        if inplace:
            self.embeddings = emb
            self.normalized = True
            return self
        return FeatureBank(
            embeddings=emb,
            metadata=self.metadata.copy(),
            name=self.name,
            normalized=True,
            extra_info=dict(self.extra_info),
        )

    def subset(self, mask: Union[np.ndarray, Sequence[bool], pd.Series], name: Optional[str] = None) -> "FeatureBank":
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 1 or len(mask_arr) != len(self):
            raise ValueError(f"Mask must have shape [N] with N={len(self)}, got {mask_arr.shape}.")
        return FeatureBank(
            embeddings=self.embeddings[mask_arr],
            metadata=self.metadata.loc[mask_arr].reset_index(drop=True),
            name=self.name if name is None else name,
            normalized=self.normalized,
            extra_info=dict(self.extra_info),
        )

    def filter_normal_only(self, label_column: str = "label") -> "FeatureBank":
        if label_column not in self.metadata.columns:
            return self.copy(name=f"{self.name}_normal")
        label_s = self.metadata[label_column].astype(str).str.lower()
        mask = label_s != "anomaly"
        return self.subset(mask.to_numpy(), name=f"{self.name}_normal")

    def filter_by_machine(self, machine: str) -> "FeatureBank":
        if "machine" not in self.metadata.columns:
            raise ValueError("Metadata does not contain a 'machine' column.")
        mask = self.metadata["machine"].astype(str) == str(machine)
        return self.subset(mask.to_numpy(), name=f"{self.name}_{machine}")

    def filter_by_domain(self, domain: str) -> "FeatureBank":
        if "domain" not in self.metadata.columns:
            raise ValueError("Metadata does not contain a 'domain' column.")
        mask = self.metadata["domain"].astype(str).str.lower() == str(domain).lower()
        return self.subset(mask.to_numpy(), name=f"{self.name}_{domain}")

    def split_by_machine(self) -> Dict[str, "FeatureBank"]:
        if "machine" not in self.metadata.columns:
            raise ValueError("Metadata does not contain a 'machine' column.")
        out: Dict[str, FeatureBank] = {}
        for machine in self.metadata["machine"].dropna().astype(str).unique():
            out[str(machine)] = self.filter_by_machine(str(machine))
        return out

    def split_by_machine_domain(self) -> Dict[str, Dict[str, "FeatureBank"]]:
        if "machine" not in self.metadata.columns:
            raise ValueError("Metadata does not contain a 'machine' column.")
        if "domain" not in self.metadata.columns:
            raise ValueError("Metadata does not contain a 'domain' column.")

        out: Dict[str, Dict[str, FeatureBank]] = {}
        for machine, machine_bank in self.split_by_machine().items():
            out[machine] = {}
            for domain in machine_bank.metadata["domain"].dropna().astype(str).unique():
                out[machine][str(domain)] = machine_bank.filter_by_domain(str(domain))
        return out

    def to_dataframe(self, include_embedding_columns: bool = False, embedding_prefix: str = "emb_") -> pd.DataFrame:
        df = self.metadata.copy()
        if include_embedding_columns:
            emb_cols = {f"{embedding_prefix}{i}": self.embeddings[:, i] for i in range(self.dim)}
            emb_df = pd.DataFrame(emb_cols)
            df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
        return df

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "name": self.name,
            "num_samples": len(self),
            "embedding_dim": self.dim,
            "normalized": bool(self.normalized),
            "metadata_columns": list(self.metadata.columns),
            "num_machines": len(self.machines),
            "machines": self.machines,
            "domains": self.domains,
        }

        if "label" in self.metadata.columns:
            counts = self.metadata["label"].fillna("<none>").astype(str).value_counts().to_dict()
            summary["label_counts"] = {str(k): int(v) for k, v in counts.items()}
        if "machine" in self.metadata.columns:
            counts = self.metadata["machine"].fillna("<none>").astype(str).value_counts().to_dict()
            summary["machine_counts"] = {str(k): int(v) for k, v in counts.items()}
        if "domain" in self.metadata.columns:
            counts = self.metadata["domain"].fillna("<none>").astype(str).value_counts().to_dict()
            summary["domain_counts"] = {str(k): int(v) for k, v in counts.items()}
        if self.extra_info:
            summary["extra_info"] = dict(self.extra_info)
        return summary

    def save(self, save_dir: Union[str, Path], overwrite: bool = True) -> Path:
        """Save the bank as:
        - embeddings.npy
        - metadata.csv
        - manifest.json
        """
        save_dir = Path(save_dir)
        if save_dir.exists() and not overwrite:
            raise FileExistsError(f"save_dir already exists: {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "embeddings.npy", self.embeddings)
        self.metadata.to_csv(save_dir / "metadata.csv", index=False)

        manifest = {
            "name": self.name,
            "normalized": bool(self.normalized),
            "shape": list(self.embeddings.shape),
            "metadata_columns": list(self.metadata.columns),
            "extra_info": self.extra_info,
        }
        with open(save_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return save_dir

    @classmethod
    def load(cls, save_dir: Union[str, Path]) -> "FeatureBank":
        save_dir = Path(save_dir)
        emb_path = save_dir / "embeddings.npy"
        meta_path = save_dir / "metadata.csv"
        manifest_path = save_dir / "manifest.json"

        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        embeddings = np.load(emb_path)
        metadata = pd.read_csv(meta_path)

        name = save_dir.name
        normalized = False
        extra_info: Dict[str, Any] = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            name = manifest.get("name", name)
            normalized = bool(manifest.get("normalized", False))
            extra_info = dict(manifest.get("extra_info", {}))

        return cls(
            embeddings=embeddings,
            metadata=metadata,
            name=name,
            normalized=normalized,
            extra_info=extra_info,
        )


# -----------------------------------------------------------------------------
# Accumulator for batch-wise extraction
# -----------------------------------------------------------------------------
class FeatureBankAccumulator:
    """Accumulate embeddings + metadata across batches, then finalize a bank.

    This is especially useful inside validation/test loops where features are
    extracted batch-by-batch.
    """

    def __init__(self, name: str = "feature_bank") -> None:
        self.name = str(name)
        self._embedding_chunks: List[np.ndarray] = []
        self._metadata_chunks: List[pd.DataFrame] = []

    def __len__(self) -> int:
        return int(sum(len(chunk) for chunk in self._embedding_chunks))

    def reset(self) -> None:
        self._embedding_chunks.clear()
        self._metadata_chunks.clear()

    def append(self, embeddings: ArrayLike, metadata: Any) -> None:
        emb = _to_numpy_2d(embeddings)
        meta_df = _normalize_batch_metadata(metadata, expected_len=len(emb))
        meta_df = _ensure_optional_columns(meta_df)
        self._embedding_chunks.append(emb)
        self._metadata_chunks.append(meta_df)

    def finalize(
        self,
        name: Optional[str] = None,
        l2_normalize: bool = False,
        extra_info: Optional[Mapping[str, Any]] = None,
        reset: bool = False,
    ) -> FeatureBank:
        if len(self._embedding_chunks) == 0:
            raise ValueError("FeatureBankAccumulator is empty. Append at least one batch first.")

        embeddings = np.concatenate(self._embedding_chunks, axis=0)
        metadata = pd.concat(self._metadata_chunks, axis=0, ignore_index=True)
        bank = FeatureBank(
            embeddings=embeddings,
            metadata=metadata,
            name=self.name if name is None else str(name),
            normalized=False,
            extra_info=dict(extra_info or {}),
        )
        if l2_normalize:
            bank = bank.l2_normalize(inplace=False)
        if reset:
            self.reset()
        return bank


# -----------------------------------------------------------------------------
# Convenience constructors
# -----------------------------------------------------------------------------

def build_feature_bank(
    embeddings: ArrayLike,
    metadata: Optional[MetadataLike] = None,
    metadata_csv_path: Optional[Union[str, Path]] = None,
    name: str = "feature_bank",
    l2_normalize: bool = False,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> FeatureBank:
    """Build a :class:`FeatureBank` from embeddings and aligned metadata."""
    emb = _to_numpy_2d(embeddings)
    meta_df = _resolve_metadata(metadata=metadata, csv_path=metadata_csv_path, split_name="metadata")
    if len(emb) != len(meta_df):
        raise ValueError(
            f"Embedding / metadata length mismatch: len(embeddings)={len(emb)} vs len(metadata)={len(meta_df)}."
        )
    bank = FeatureBank(
        embeddings=emb,
        metadata=meta_df,
        name=name,
        normalized=False,
        extra_info=dict(extra_info or {}),
    )
    if l2_normalize:
        bank = bank.l2_normalize(inplace=False)
    return bank



def concatenate_feature_banks(
    banks: Sequence[FeatureBank],
    name: str = "concatenated_bank",
) -> FeatureBank:
    """Concatenate multiple feature banks along the sample dimension."""
    banks = list(banks)
    if len(banks) == 0:
        raise ValueError("At least one bank is required.")

    dim = banks[0].dim
    normalized = all(bank.normalized for bank in banks)
    for bank in banks[1:]:
        if bank.dim != dim:
            raise ValueError(f"All banks must have the same embedding dimension. Got {dim} and {bank.dim}.")

    embeddings = np.concatenate([bank.embeddings for bank in banks], axis=0)
    metadata = pd.concat([bank.metadata for bank in banks], axis=0, ignore_index=True)
    extra_info = {
        "source_banks": [bank.name for bank in banks],
    }
    return FeatureBank(
        embeddings=embeddings,
        metadata=metadata,
        name=name,
        normalized=normalized,
        extra_info=extra_info,
    )


__all__ = [
    "FeatureBank",
    "FeatureBankAccumulator",
    "build_feature_bank",
    "concatenate_feature_banks",
]
