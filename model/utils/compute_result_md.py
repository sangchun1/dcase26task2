"""Mahalanobis-distance scoring utilities for ASD separator embeddings.

This module is the Mahalanobis-distance counterpart of the project's existing
``compute_result.py`` / ``compute_result_ldn.py`` files.

Why this file exists
--------------------
For the source-separation proxy pipeline, the first anomaly scorer we want is a
simple and strong density-based method:

1. fit a Gaussian to *normal* train embeddings,
2. compute Mahalanobis distance for each test embedding,
3. use that distance as the anomaly score.

This file supports two common ASD evaluation modes:

- **global-per-machine MD**
  Fit a single Gaussian per machine using all normal train embeddings of that
  machine.
- **source/target-min MD** (default)
  Fit *two* Gaussians per machine (source and target domain) and score each
  test sample by the smaller of the two distances. This mirrors the existing
  source/target-aware evaluation style in the current project.

Primary entry point
-------------------
``compute_result_md`` is designed to be easy to plug into the upcoming
``ssmodule_sep.py`` validation/test loop.

Validation usage
----------------
>>> machine_results, mean_auc_s, mean_auc_t, mean_p_auc, final_score = compute_result_md(
...     result_train=train_embeddings,
...     result_test=val_embeddings,
...     train_csv_path="data/dev_train.csv",
...     test_csv_path="data/dev_test.csv",
...     test=False,
... )

Test / eval usage
-----------------
>>> anomaly_scores = compute_result_md(
...     result_train=train_embeddings,
...     result_test=eval_embeddings,
...     train_csv_path="data/dev_train.csv",
...     test_csv_path="data/eval.csv",
...     test=True,
... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ArrayLike = Union[np.ndarray, Sequence[float]]

EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _to_numpy(x: Union[np.ndarray, "object"]) -> np.ndarray:
    """Convert torch / numpy inputs to a contiguous float32 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {x.shape}.")
    return np.ascontiguousarray(x)



def _resolve_metadata(
    metadata: Optional[Union[pd.DataFrame, Mapping, Sequence[Mapping]]] = None,
    csv_path: Optional[str] = None,
    split_name: str = "metadata",
) -> pd.DataFrame:
    """Resolve metadata from a dataframe-like object or a CSV path."""
    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            df = metadata.copy()
        else:
            df = pd.DataFrame(metadata)
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(
            f"{split_name} was not provided. Pass either `{split_name}` as a DataFrame-like object "
            f"or `{split_name[:-8] if split_name.endswith('_metadata') else split_name}_csv_path`."
        )

    return df.reset_index(drop=True)



def _validate_alignment(embeddings: np.ndarray, metadata: pd.DataFrame, split_name: str) -> None:
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding / metadata length mismatch for {split_name}: "
            f"len(embeddings)={len(embeddings)} vs len(metadata)={len(metadata)}."
        )



def _normalize_binary_labels(labels: Iterable) -> np.ndarray:
    """Map labels to {0,1} where possible.

    Accepted values include:
    - strings: "normal", "anomaly"
    - ints / bools already in {0,1}
    """
    s = pd.Series(list(labels))

    if s.dtype == object:
        lowered = s.astype(str).str.lower()
        mapped = lowered.map({"normal": 0, "anomaly": 1})
        if mapped.isna().any():
            raise ValueError(
                "Could not map labels to binary values. Supported string labels are 'normal' and 'anomaly'."
            )
        return mapped.to_numpy(dtype=np.int64)

    arr = s.to_numpy()
    unique = np.unique(arr)
    if np.all(np.isin(unique, [0, 1])):
        return arr.astype(np.int64)

    raise ValueError(f"Unsupported label values for binary evaluation: {unique}.")



def _harmonic_mean(values: Sequence[float], eps: float = EPS) -> float:
    valid = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=np.float64)
    if valid.size == 0:
        return float("nan")
    valid = np.clip(valid, eps, None)
    return float(valid.size / np.sum(1.0 / valid))



def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))



def _safe_pauc(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores, max_fpr=max_fpr))


# -----------------------------------------------------------------------------
# Mahalanobis core
# -----------------------------------------------------------------------------
@dataclass
class GaussianBank:
    """Single Gaussian reference bank for Mahalanobis scoring."""

    mean: np.ndarray
    inv_cov: np.ndarray
    covariance_type: str
    n_samples: int


class MahalanobisScorer:
    """Fit a Gaussian reference distribution and score by Mahalanobis distance."""

    def __init__(self, regularization: float = 1e-5, covariance_type: str = "full") -> None:
        if covariance_type not in {"full", "diag"}:
            raise ValueError("covariance_type must be one of {'full', 'diag'}.")
        self.regularization = float(regularization)
        self.covariance_type = covariance_type
        self.bank: Optional[GaussianBank] = None

    def fit(self, embeddings: ArrayLike) -> "MahalanobisScorer":
        x = _to_numpy(embeddings)
        if len(x) == 0:
            raise ValueError("Cannot fit MahalanobisScorer on an empty embedding set.")

        mean = x.mean(axis=0)
        centered = x - mean[None, :]
        dim = x.shape[1]

        if self.covariance_type == "diag":
            var = centered.var(axis=0) + self.regularization
            inv_cov = np.diag(1.0 / np.clip(var, EPS, None)).astype(np.float32)
        else:
            if len(x) == 1:
                cov = np.eye(dim, dtype=np.float32) * self.regularization
            else:
                cov = np.cov(centered, rowvar=False).astype(np.float32)
                cov = np.atleast_2d(cov)
                cov += np.eye(dim, dtype=np.float32) * self.regularization
            inv_cov = np.linalg.pinv(cov).astype(np.float32)

        self.bank = GaussianBank(
            mean=mean.astype(np.float32),
            inv_cov=inv_cov,
            covariance_type=self.covariance_type,
            n_samples=len(x),
        )
        return self

    def score(self, queries: ArrayLike) -> np.ndarray:
        if self.bank is None:
            raise RuntimeError("MahalanobisScorer is not fitted yet.")

        q = _to_numpy(queries)
        diff = q - self.bank.mean[None, :]
        scores = np.einsum("bi,ij,bj->b", diff, self.bank.inv_cov, diff, optimize=True)
        return scores.astype(np.float32)


# -----------------------------------------------------------------------------
# Bank construction helpers
# -----------------------------------------------------------------------------

def _fit_machine_bank(
    machine_embeddings: np.ndarray,
    regularization: float,
    covariance_type: str,
) -> MahalanobisScorer:
    scorer = MahalanobisScorer(regularization=regularization, covariance_type=covariance_type)
    scorer.fit(machine_embeddings)
    return scorer



def _build_banks(
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    domain_strategy: str = "source_target_min",
    regularization: float = 1e-5,
    covariance_type: str = "full",
) -> Dict[str, Dict[str, MahalanobisScorer]]:
    """Build per-machine Mahalanobis reference banks.

    Parameters
    ----------
    domain_strategy:
        - ``global``: one Gaussian per machine using all train rows.
        - ``source_target_min``: separate source / target banks per machine,
          and later score by the smaller distance.
    """
    if "machine" not in train_df.columns:
        raise ValueError("train metadata must contain a 'machine' column.")

    if domain_strategy not in {"global", "source_target_min"}:
        raise ValueError("domain_strategy must be one of {'global', 'source_target_min'}.")

    banks: Dict[str, Dict[str, MahalanobisScorer]] = {}

    for machine in train_df["machine"].astype(str).unique():
        machine_mask = train_df["machine"].astype(str) == machine
        machine_train = train_embeddings[machine_mask.to_numpy()]
        machine_meta = train_df.loc[machine_mask].reset_index(drop=True)

        if len(machine_train) == 0:
            continue

        banks[machine] = {}

        if domain_strategy == "global" or "domain" not in machine_meta.columns:
            banks[machine]["global"] = _fit_machine_bank(
                machine_train,
                regularization=regularization,
                covariance_type=covariance_type,
            )
            continue

        # source / target split
        for domain in ["source", "target"]:
            dmask = machine_meta["domain"].astype(str).str.lower() == domain
            x = machine_train[dmask.to_numpy()]
            if len(x) == 0:
                continue
            banks[machine][domain] = _fit_machine_bank(
                x,
                regularization=regularization,
                covariance_type=covariance_type,
            )

        # Fallback if one of the domain splits is missing.
        if len(banks[machine]) == 0:
            banks[machine]["global"] = _fit_machine_bank(
                machine_train,
                regularization=regularization,
                covariance_type=covariance_type,
            )

    return banks



def _score_machine_sample(
    embedding: np.ndarray,
    machine: str,
    banks: Dict[str, Dict[str, MahalanobisScorer]],
    domain_strategy: str = "source_target_min",
) -> float:
    if machine not in banks:
        raise KeyError(f"No Mahalanobis bank found for machine={machine!r}.")

    machine_banks = banks[machine]
    x = embedding.reshape(1, -1)

    if domain_strategy == "global" or "global" in machine_banks:
        return float(machine_banks.get("global", next(iter(machine_banks.values()))).score(x)[0])

    # source_target_min
    candidates: List[float] = []
    for domain in ["source", "target"]:
        if domain in machine_banks:
            candidates.append(float(machine_banks[domain].score(x)[0]))

    if not candidates:
        # Fallback for robustness.
        candidates.append(float(next(iter(machine_banks.values())).score(x)[0]))

    return float(np.min(candidates))



def compute_md_scores(
    train_embeddings: ArrayLike,
    test_embeddings: ArrayLike,
    train_metadata: Union[pd.DataFrame, Mapping, Sequence[Mapping]],
    test_metadata: Union[pd.DataFrame, Mapping, Sequence[Mapping]],
    regularization: float = 1e-5,
    covariance_type: str = "full",
    domain_strategy: str = "source_target_min",
) -> np.ndarray:
    """Compute anomaly scores for test embeddings using Mahalanobis distance.

    Returns
    -------
    np.ndarray
        Shape ``[N_test]``.
    """
    train_x = _to_numpy(train_embeddings)
    test_x = _to_numpy(test_embeddings)
    train_df = _resolve_metadata(train_metadata, split_name="train_metadata")
    test_df = _resolve_metadata(test_metadata, split_name="test_metadata")

    _validate_alignment(train_x, train_df, "train")
    _validate_alignment(test_x, test_df, "test")

    banks = _build_banks(
        train_embeddings=train_x,
        train_df=train_df,
        domain_strategy=domain_strategy,
        regularization=regularization,
        covariance_type=covariance_type,
    )

    scores = np.zeros(len(test_x), dtype=np.float32)
    machines = test_df["machine"].astype(str).to_numpy()
    for i, machine in enumerate(machines):
        scores[i] = _score_machine_sample(
            embedding=test_x[i],
            machine=machine,
            banks=banks,
            domain_strategy=domain_strategy,
        )
    return scores


# -----------------------------------------------------------------------------
# Validation summary
# -----------------------------------------------------------------------------

def summarize_md_validation(
    test_scores: ArrayLike,
    test_metadata: Union[pd.DataFrame, Mapping, Sequence[Mapping]],
    max_fpr: float = 0.1,
) -> Tuple[Dict[str, Dict[str, float]], float, float, float, float]:
    """Compute DCASE-style validation summary from anomaly scores.

    Returns
    -------
    machine_results, mean_auc_source, mean_auc_target, mean_p_auc, final_score
    """
    scores = np.asarray(test_scores, dtype=np.float32)
    test_df = _resolve_metadata(test_metadata, split_name="test_metadata")
    _validate_alignment(scores.reshape(-1, 1), test_df, "test")

    if "machine" not in test_df.columns:
        raise ValueError("test metadata must contain a 'machine' column.")
    if "label" not in test_df.columns:
        raise ValueError("test metadata must contain a 'label' column for validation.")

    labels = _normalize_binary_labels(test_df["label"])
    test_df = test_df.copy()
    test_df["_label_bin"] = labels
    test_df["_score"] = scores

    has_domain = "domain" in test_df.columns
    machine_results: Dict[str, Dict[str, float]] = {}

    auc_s_list: List[float] = []
    auc_t_list: List[float] = []
    pauc_list: List[float] = []

    for machine in test_df["machine"].astype(str).unique():
        mdf = test_df.loc[test_df["machine"].astype(str) == machine].reset_index(drop=True)
        y = mdf["_label_bin"].to_numpy(dtype=np.int64)
        s = mdf["_score"].to_numpy(dtype=np.float32)

        machine_pauc = _safe_pauc(y, s, max_fpr=max_fpr)
        pauc_list.append(machine_pauc)

        auc_source = float("nan")
        auc_target = float("nan")

        if has_domain:
            for domain_name, key in [("source", "AUC_source"), ("target", "AUC_target")]:
                ddf = mdf.loc[mdf["domain"].astype(str).str.lower() == domain_name]
                if len(ddf) == 0:
                    value = float("nan")
                else:
                    value = _safe_auc(
                        ddf["_label_bin"].to_numpy(dtype=np.int64),
                        ddf["_score"].to_numpy(dtype=np.float32),
                    )
                if domain_name == "source":
                    auc_source = value
                    auc_s_list.append(value)
                else:
                    auc_target = value
                    auc_t_list.append(value)
        else:
            auc_source = _safe_auc(y, s)
            auc_s_list.append(auc_source)

        machine_results[machine] = {
            "AUC_source": float(auc_source),
            "AUC_target": float(auc_target),
            "pAUC": float(machine_pauc),
            "hmean": _harmonic_mean([auc_source, auc_target, machine_pauc]),
            "num_samples": int(len(mdf)),
        }

    mean_auc_source = float(np.nanmean(auc_s_list)) if len(auc_s_list) > 0 else float("nan")
    mean_auc_target = float(np.nanmean(auc_t_list)) if len(auc_t_list) > 0 else float("nan")
    mean_p_auc = float(np.nanmean(pauc_list)) if len(pauc_list) > 0 else float("nan")
    final_score = _harmonic_mean([mean_auc_source, mean_auc_target, mean_p_auc])

    return machine_results, mean_auc_source, mean_auc_target, mean_p_auc, final_score


# -----------------------------------------------------------------------------
# High-level project-style wrapper
# -----------------------------------------------------------------------------

def compute_result_md(
    result_train: ArrayLike,
    result_test: ArrayLike,
    regularization: float = 1e-5,
    covariance_type: str = "full",
    domain_strategy: str = "source_target_min",
    train_metadata: Optional[Union[pd.DataFrame, Mapping, Sequence[Mapping]]] = None,
    test_metadata: Optional[Union[pd.DataFrame, Mapping, Sequence[Mapping]]] = None,
    train_csv_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    test: bool = False,
    max_fpr: float = 0.1,
):
    """Project-style wrapper for Mahalanobis anomaly scoring.

    This function intentionally mirrors the return style of the existing
    ``compute_result`` utilities:

    - validation mode (``test=False``):
      returns ``machine_results, mean_auc_source, mean_auc_target, mean_p_auc, final_score``
    - eval mode (``test=True``):
      returns ``anomaly_scores`` only

    Parameters
    ----------
    result_train, result_test:
        Train / validation(eval) embeddings of shape ``[N, D]``.
    regularization:
        Diagonal loading term added to the covariance.
    covariance_type:
        ``'full'`` or ``'diag'``.
    domain_strategy:
        ``'source_target_min'`` or ``'global'``.
    train_metadata, test_metadata:
        Optional dataframe-like metadata. Use these when available.
    train_csv_path, test_csv_path:
        CSV fallbacks when metadata dataframes are not passed directly.
    test:
        If ``False``, compute validation AUC / pAUC summary.
        If ``True``, return raw anomaly scores only.
    max_fpr:
        pAUC maximum false-positive rate.
    """
    train_df = _resolve_metadata(train_metadata, train_csv_path, split_name="train_metadata")
    test_df = _resolve_metadata(test_metadata, test_csv_path, split_name="test_metadata")

    anomaly_scores = compute_md_scores(
        train_embeddings=result_train,
        test_embeddings=result_test,
        train_metadata=train_df,
        test_metadata=test_df,
        regularization=regularization,
        covariance_type=covariance_type,
        domain_strategy=domain_strategy,
    )

    if test:
        return anomaly_scores

    return summarize_md_validation(
        test_scores=anomaly_scores,
        test_metadata=test_df,
        max_fpr=max_fpr,
    )


# -----------------------------------------------------------------------------
# Lightweight convenience utilities for analysis / debugging
# -----------------------------------------------------------------------------

def attach_scores_to_metadata(
    metadata: Union[pd.DataFrame, Mapping, Sequence[Mapping]],
    scores: ArrayLike,
    score_column: str = "anomaly_score",
) -> pd.DataFrame:
    """Return a copy of metadata with an appended anomaly-score column."""
    df = _resolve_metadata(metadata, split_name="metadata")
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(df) != len(scores):
        raise ValueError(f"Length mismatch: len(metadata)={len(df)} vs len(scores)={len(scores)}.")
    out = df.copy()
    out[score_column] = scores
    return out


__all__ = [
    "GaussianBank",
    "MahalanobisScorer",
    "compute_md_scores",
    "summarize_md_validation",
    "compute_result_md",
    "attach_scores_to_metadata",
]
