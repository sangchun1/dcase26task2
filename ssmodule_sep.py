"""Lightning module for source-separation proxy training and ASD scoring.

This module is the separation-based counterpart of the original ``ssmodule.py``.
It keeps the project workflow familiar while changing the core proxy task:

    synthetic mixture -> separator training -> separator feature extraction
    -> Mahalanobis anomaly scoring

Design goals
------------
1. Train a separator on ``(mix_wave, target_wave)`` pairs from
   ``SeparationDataset``.
2. Use the trained separator + ``SeparatorFeatureHead`` to extract embeddings
   from original waveforms.
3. Score validation / test samples with Mahalanobis distance using a normal
   train feature bank.
4. Save top-k checkpoints in the same style as the existing project.

Recommended first configuration
-------------------------------
- separator backbone: ResUNet
- frontend input/target representation: magnitude
- output_mode: mask
- loss: L1(spec) + 0.1 * neg-SI-SDR
- scorer: Mahalanobis distance
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import lightning.pytorch as pl

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.utils.optimizer import create_optimizer_scheduler
from model.data.separation_dataset import SeparationDataset, separation_collate_fn
from model.data.raw_wave_dataset import RawWaveDataset, raw_wave_collate_fn
from model.separator.frontend import STFTFrontend, FrontendConfig
from model.separator.resunet_separator import ResUNetSeparator, ResUNetSeparatorConfig
from model.separator.feature_head import SeparatorFeatureHead, FeatureHeadConfig
from model.utils.sep_loss import SeparationLoss, SeparationLossConfig, build_default_separator_loss
from model.utils.sep_metric import (
    SeparationMetricConfig,
    SeparationMetricTracker,
    build_default_separator_metric_config,
    compute_separator_metrics,
)
from model.utils.compute_result_md import compute_result_md
from model.utils.feature_bank import FeatureBank, FeatureBankAccumulator


class ssmodule_sep(pl.LightningModule):
    """Source-separation proxy LightningModule for ASD.

    Expected high-level kwargs
    --------------------------
    Data:
    - train_path
    - val_path
    - test_path
    - dev_train_path

    Training:
    - batch_size, num_workers, epochs
    - optimizer, scheduler, max_lr, min_lr, warmup_epochs, restart_period, ...

    Separation dataset:
    - sample_rate
    - segment_seconds
    - snr_min_db, snr_max_db
    - interference_mode
    - noise_csv_path (optional)

    Scoring:
    - md_regularization
    - md_covariance_type
    - md_domain_strategy
    - max_fpr

    The module is intentionally permissive and fills many defaults when the
    separation-specific arguments are not provided.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = dict(kwargs)
        self.save_hyperparameters(ignore=[])

        # ------------------------------------------------------------------
        # Core components
        # ------------------------------------------------------------------
        self.frontend = self._build_frontend()
        self.separator = self._build_separator()
        self.feature_head = self._build_feature_head()
        self.loss_fn = self._build_loss()
        self.metric_cfg = self._build_metric_config()
        self.metric_tracker = SeparationMetricTracker()

        # ------------------------------------------------------------------
        # Runtime buffers / accumulators
        # ------------------------------------------------------------------
        self.validation_feature_accumulator = FeatureBankAccumulator(name="validation_features")
        self.test_feature_accumulator = FeatureBankAccumulator(name="test_features")

        self.best_checkpoints: List[Tuple[float, str]] = []
        self.save_top_k = int(self.kwargs.get("save_top_k", 3))
        self.best_final_score = float("-inf")

        self.md_best_state: Dict[str, Any] = {
            "regularization": float(self.kwargs.get("md_regularization", 1e-5)),
            "covariance_type": str(self.kwargs.get("md_covariance_type", "full")),
            "domain_strategy": str(self.kwargs.get("md_domain_strategy", "source_target_min")),
        }

        devices = str(self.kwargs.get("devices", "0"))
        self.distributed_mode = len([d for d in devices.split(",") if d.strip() != ""]) > 1

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_frontend(self) -> STFTFrontend:
        cfg = FrontendConfig(
            sample_rate=int(self.kwargs.get("sample_rate", 16000)),
            n_fft=int(self.kwargs.get("n_fft", 1024)),
            hop_length=int(self.kwargs.get("hop_length", 512)),
            win_length=self.kwargs.get("win_length", None),
            window=str(self.kwargs.get("window", "hann")),
            center=bool(self.kwargs.get("stft_center", True)),
            normalized=bool(self.kwargs.get("stft_normalized", False)),
            input_representation=str(self.kwargs.get("input_representation", "magnitude")),
            target_representation=str(self.kwargs.get("target_representation", "magnitude")),
            mag_eps=float(self.kwargs.get("mag_eps", 1e-8)),
        )
        return STFTFrontend(cfg)

    def _build_separator(self) -> ResUNetSeparator:
        sep_cfg = ResUNetSeparatorConfig(
            in_channels=int(self.kwargs.get("sep_in_channels", 1)),
            encoder_channels=tuple(self.kwargs.get("sep_encoder_channels", (32, 64, 128))),
            bottleneck_channels=self.kwargs.get("sep_bottleneck_channels", None),
            stem_channels=self.kwargs.get("sep_stem_channels", None),
            num_encoder_res_blocks=int(self.kwargs.get("sep_num_encoder_res_blocks", 2)),
            num_bottleneck_res_blocks=int(self.kwargs.get("sep_num_bottleneck_res_blocks", 2)),
            num_decoder_res_blocks=int(self.kwargs.get("sep_num_decoder_res_blocks", 2)),
            downsample_mode=str(self.kwargs.get("sep_downsample_mode", "conv")),
            upsample_mode=str(self.kwargs.get("sep_upsample_mode", "bilinear")),
            output_mode=str(self.kwargs.get("sep_output_mode", "mask")),
            output_activation=self.kwargs.get("sep_output_activation", None),
            return_all_encoder_features=bool(self.kwargs.get("sep_return_all_encoder_features", True)),
            return_decoder_features=bool(self.kwargs.get("sep_return_decoder_features", False)),
            asd_feature_source=str(self.kwargs.get("sep_asd_feature_source", "encoder_bottleneck")),
        )
        return ResUNetSeparator(sep_cfg)

    def _build_feature_head(self) -> SeparatorFeatureHead:
        head_cfg = FeatureHeadConfig(
            pooling=str(self.kwargs.get("feature_pooling", "mean")),
            projection_dim=self.kwargs.get("feature_projection_dim", None),
            aggregation=str(self.kwargs.get("feature_aggregation", "concat")),
            dropout=float(self.kwargs.get("feature_dropout", 0.0)),
            use_layernorm=bool(self.kwargs.get("feature_use_layernorm", False)),
            l2_normalize=bool(self.kwargs.get("feature_l2_normalize", False)),
            detach_feature_maps=bool(self.kwargs.get("feature_detach_maps", False)),
        )
        return SeparatorFeatureHead(head_cfg)

    def _build_loss(self) -> SeparationLoss:
        if any(k in self.kwargs for k in [
            "l1_spec_weight",
            "mse_spec_weight",
            "spectral_convergence_weight",
            "sisdr_weight",
            "mask_l1_weight",
            "mask_bce_weight",
        ]):
            cfg = SeparationLossConfig(
                l1_spec_weight=float(self.kwargs.get("l1_spec_weight", 1.0)),
                mse_spec_weight=float(self.kwargs.get("mse_spec_weight", 0.0)),
                spectral_convergence_weight=float(self.kwargs.get("spectral_convergence_weight", 0.0)),
                sisdr_weight=float(self.kwargs.get("sisdr_weight", 0.1)),
                mask_l1_weight=float(self.kwargs.get("mask_l1_weight", 0.0)),
                mask_bce_weight=float(self.kwargs.get("mask_bce_weight", 0.0)),
                zero_mean_waveform=bool(self.kwargs.get("zero_mean_waveform", True)),
                reduction=str(self.kwargs.get("sep_loss_reduction", "mean")),
            )
            return SeparationLoss(cfg)
        return build_default_separator_loss()

    def _build_metric_config(self) -> SeparationMetricConfig:
        if any(k in self.kwargs for k in [
            "metric_compute_sisdr",
            "metric_compute_sisdri",
            "metric_compute_snr",
            "metric_compute_spec_l1",
            "metric_compute_spec_mse",
            "metric_compute_spectral_convergence",
            "metric_compute_mask_l1",
        ]):
            return SeparationMetricConfig(
                compute_sisdr=bool(self.kwargs.get("metric_compute_sisdr", True)),
                compute_sisdri=bool(self.kwargs.get("metric_compute_sisdri", True)),
                compute_snr=bool(self.kwargs.get("metric_compute_snr", False)),
                compute_spec_l1=bool(self.kwargs.get("metric_compute_spec_l1", True)),
                compute_spec_mse=bool(self.kwargs.get("metric_compute_spec_mse", False)),
                compute_spectral_convergence=bool(self.kwargs.get("metric_compute_spectral_convergence", False)),
                compute_mask_l1=bool(self.kwargs.get("metric_compute_mask_l1", False)),
                zero_mean_sisdr=bool(self.kwargs.get("metric_zero_mean_sisdr", True)),
                detach=bool(self.kwargs.get("metric_detach", True)),
            )
        return build_default_separator_metric_config()

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def _make_separation_dataloader(
        self,
        csv_path: str,
        *,
        shuffle: bool,
        drop_last: bool,
        fixed_snr_db: Optional[float] = None,
    ) -> DataLoader:
        dataset = SeparationDataset(
            csv_path=csv_path,
            sample_rate=int(self.kwargs.get("sample_rate", 16000)),
            segment_seconds=float(self.kwargs.get("segment_seconds", 2.0)),
            snr_min_db=float(self.kwargs.get("snr_min_db", -5.0)),
            snr_max_db=float(self.kwargs.get("snr_max_db", 5.0)),
            interference_mode=str(self.kwargs.get("interference_mode", "other_machine")),
            fixed_snr_db=fixed_snr_db,
            noise_csv_path=self.kwargs.get("noise_csv_path", None),
            noise_paths=self.kwargs.get("noise_paths", None),
            filter_normal_only=bool(self.kwargs.get("filter_normal_only", True)),
            seed=self.kwargs.get("seed", None),
            return_realized_snr=bool(self.kwargs.get("return_realized_snr", True)),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.kwargs.get("batch_size", 16)),
            num_workers=int(self.kwargs.get("num_workers", 4)),
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=separation_collate_fn,
            pin_memory=bool(self.kwargs.get("pin_memory", True)),
        )

    def _make_raw_dataloader(
        self,
        csv_path: str,
        *,
        filter_normal_only: bool,
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = RawWaveDataset(
            csv_path=csv_path,
            sample_rate=int(self.kwargs.get("sample_rate", 16000)),
            filter_normal_only=filter_normal_only,
            segment_seconds=self.kwargs.get("extract_segment_seconds", None),
            segment_mode=str(self.kwargs.get("extract_segment_mode", "center")),
            return_numpy=False,
            seed=self.kwargs.get("seed", None),
            zero_mean=bool(self.kwargs.get("extract_zero_mean", False)),
            peak_normalize=bool(self.kwargs.get("extract_peak_normalize", False)),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.kwargs.get("extract_batch_size", 1)),
            num_workers=int(self.kwargs.get("num_workers", 4)),
            shuffle=shuffle,
            drop_last=False,
            collate_fn=raw_wave_collate_fn,
            pin_memory=bool(self.kwargs.get("pin_memory", True)),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_separation_dataloader(
            self.kwargs["train_path"],
            shuffle=True,
            drop_last=True,
            fixed_snr_db=self.kwargs.get("train_fixed_snr_db", None),
        )

    def val_dataloader(self) -> DataLoader:
        return self._make_raw_dataloader(
            self.kwargs["val_path"],
            filter_normal_only=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._make_raw_dataloader(
            self.kwargs["test_path"],
            filter_normal_only=False,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # Forward / helper routines
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_wave_from_metadata(batch: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in batch.items()
            if key not in {"wave", "wave_lengths"}
        }

    @staticmethod
    def _tensor_to_scalar_dict(metrics: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                out[key] = float(value.detach().cpu().item())
            else:
                out[key] = float(value)
        return out

    def _run_separator(self, mix_wave: torch.Tensor) -> Dict[str, Any]:
        mix_spec, mix_aux = self.frontend.wave_to_input_spec(mix_wave)
        sep_out = self.separator(mix_spec)
        sep_out["mix_aux"] = mix_aux
        return sep_out

    def _prepare_train_batch(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        mix_wave = batch["mix_wave"].to(self.device, non_blocking=True)
        target_wave = batch["target_wave"].to(self.device, non_blocking=True)
        if mix_wave.ndim == 3 and mix_wave.shape[1] == 1:
            pass
        elif mix_wave.ndim == 2:
            mix_wave = mix_wave.unsqueeze(1)
        if target_wave.ndim == 3 and target_wave.shape[1] == 1:
            pass
        elif target_wave.ndim == 2:
            target_wave = target_wave.unsqueeze(1)
        return {"mix_wave": mix_wave, "target_wave": target_wave}

    def _build_target_mask_if_needed(self, mix_wave: torch.Tensor, target_wave: torch.Tensor) -> Optional[torch.Tensor]:
        if float(self.kwargs.get("mask_l1_weight", 0.0)) <= 0.0 and float(self.kwargs.get("mask_bce_weight", 0.0)) <= 0.0:
            return None
        return self.frontend.make_target_mask(mix_wave, target_wave)

    def _extract_embedding_tensor(self, wave: torch.Tensor) -> torch.Tensor:
        input_spec, _ = self.frontend.wave_to_input_spec(wave)
        sep_out = self.separator(input_spec)
        head_out = self.feature_head(sep_out)
        return head_out["embedding"]

    def _extract_embeddings_from_raw_batch(self, batch: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        metadata = self._strip_wave_from_metadata(batch)
        wave = batch["wave"]

        if torch.is_tensor(wave):
            wave = wave.to(self.device, non_blocking=True)
            emb = self._extract_embedding_tensor(wave)
            return emb, metadata

        # Variable-length case: wave is a list of [1, T] tensors.
        embeddings: List[torch.Tensor] = []
        for item in wave:
            item = item.to(self.device, non_blocking=True)
            emb = self._extract_embedding_tensor(item.unsqueeze(0) if item.ndim == 2 else item)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0), metadata

    def _build_feature_bank_from_loader(self, loader: DataLoader, name: str) -> FeatureBank:
        accumulator = FeatureBankAccumulator(name=name)
        self.separator.eval()
        self.feature_head.eval()
        self.frontend.eval()

        with torch.no_grad():
            for batch in loader:
                embeddings, metadata = self._extract_embeddings_from_raw_batch(batch)
                accumulator.append(embeddings, metadata)

        bank = accumulator.finalize(name=name, l2_normalize=bool(self.kwargs.get("feature_bank_l2_normalize", False)))
        return self._gather_feature_bank(bank)

    def _train_feature_bank(self) -> FeatureBank:
        dev_train_path = self.kwargs.get("dev_train_path", self.kwargs["train_path"])
        loader = self._make_raw_dataloader(dev_train_path, filter_normal_only=True, shuffle=False)
        return self._build_feature_bank_from_loader(loader, name="train_feature_bank")

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------
    def _gather_feature_bank(self, bank: FeatureBank) -> FeatureBank:
        if not (dist.is_available() and dist.is_initialized()):
            return bank

        world_size = dist.get_world_size()
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        payload = {
            "embeddings": bank.embeddings,
            "metadata": bank.metadata.to_dict(orient="list"),
            "name": bank.name,
            "normalized": bank.normalized,
            "extra_info": bank.extra_info,
        }
        dist.all_gather_object(gathered, payload)

        embeddings = np.concatenate([g["embeddings"] for g in gathered if g is not None], axis=0)
        metadata = pd.concat(
            [pd.DataFrame(g["metadata"]) for g in gathered if g is not None],
            axis=0,
            ignore_index=True,
        )
        return FeatureBank(
            embeddings=embeddings,
            metadata=metadata,
            name=bank.name,
            normalized=bank.normalized,
            extra_info=dict(bank.extra_info),
        )

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def forward(self, mix_wave: torch.Tensor) -> Dict[str, Any]:
        return self._run_separator(mix_wave)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        tensors = self._prepare_train_batch(batch)
        mix_wave = tensors["mix_wave"]
        target_wave = tensors["target_wave"]

        sep_out = self._run_separator(mix_wave)
        target_spec = self.frontend.wave_to_target_spec(target_wave)
        pred_wave = self.frontend.pred_spec_to_wave(
            sep_out["pred_spec"],
            sep_out["mix_aux"],
            length=int(target_wave.shape[-1]),
        )
        target_mask = self._build_target_mask_if_needed(mix_wave, target_wave)

        loss_dict = self.loss_fn(
            pred_spec=sep_out["pred_spec"],
            target_spec=target_spec,
            pred_wave=pred_wave,
            target_wave=target_wave,
            pred_mask=sep_out.get("pred_mask", None),
            target_mask=target_mask,
        )
        loss = loss_dict["loss"]

        self.log("train_loss", loss, batch_size=int(mix_wave.shape[0]), prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            if key == "loss":
                continue
            self.log(f"train/{key}", value, batch_size=int(mix_wave.shape[0]), sync_dist=True)

        if bool(self.kwargs.get("log_train_sep_metrics", True)):
            metrics = compute_separator_metrics(
                pred_wave=pred_wave,
                target_wave=target_wave,
                mix_wave=mix_wave,
                pred_spec=sep_out["pred_spec"],
                target_spec=target_spec,
                pred_mask=sep_out.get("pred_mask", None),
                target_mask=target_mask,
                config=self.metric_cfg,
                prefix="train/",
            )
            for key, value in metrics.items():
                self.log(key, value, batch_size=int(mix_wave.shape[0]), sync_dist=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_feature_accumulator.reset()
        self.metric_tracker.reset()

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            embeddings, metadata = self._extract_embeddings_from_raw_batch(batch)
        self.validation_feature_accumulator.append(embeddings, metadata)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.validation_feature_accumulator.reset()
            return

        val_bank_local = self.validation_feature_accumulator.finalize(
            name="validation_feature_bank",
            l2_normalize=bool(self.kwargs.get("feature_bank_l2_normalize", False)),
            reset=True,
        )
        val_bank = self._gather_feature_bank(val_bank_local)
        train_bank = self._train_feature_bank()

        regularization = float(self.kwargs.get("md_regularization", 1e-5))
        covariance_type = str(self.kwargs.get("md_covariance_type", "full"))
        domain_strategy = str(self.kwargs.get("md_domain_strategy", "source_target_min"))
        max_fpr = float(self.kwargs.get("max_fpr", 0.1))

        (
            machine_results,
            mean_auc_source,
            mean_auc_target,
            mean_p_auc,
            final_score,
        ) = compute_result_md(
            result_train=train_bank.embeddings,
            result_test=val_bank.embeddings,
            regularization=regularization,
            covariance_type=covariance_type,
            domain_strategy=domain_strategy,
            train_metadata=train_bank.metadata,
            test_metadata=val_bank.metadata,
            test=False,
            max_fpr=max_fpr,
        )

        self.log("mean_AUC_source", mean_auc_source, add_dataloader_idx=False, sync_dist=True, prog_bar=True)
        self.log("mean_AUC_target", mean_auc_target, add_dataloader_idx=False, sync_dist=True)
        self.log("mean_pAUC", mean_p_auc, add_dataloader_idx=False, sync_dist=True)
        self.log("final_score", final_score, add_dataloader_idx=False, sync_dist=True, prog_bar=True)

        if self.trainer.is_global_zero:
            self.md_best_state = {
                "regularization": regularization,
                "covariance_type": covariance_type,
                "domain_strategy": domain_strategy,
                "machine_results": machine_results,
            }
            self.update_best_checkpoints(float(final_score))

    def on_test_epoch_start(self) -> None:
        self.test_feature_accumulator.reset()

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            embeddings, metadata = self._extract_embeddings_from_raw_batch(batch)
        self.test_feature_accumulator.append(embeddings, metadata)

    def on_test_epoch_end(self):
        test_bank_local = self.test_feature_accumulator.finalize(
            name="test_feature_bank",
            l2_normalize=bool(self.kwargs.get("feature_bank_l2_normalize", False)),
            reset=True,
        )

        test_bank = self._gather_feature_bank(test_bank_local)
        train_bank = self._train_feature_bank()

        anomaly_scores = compute_result_md(
            result_train=train_bank.embeddings,
            result_test=test_bank.embeddings,
            regularization=float(self.md_best_state.get("regularization", self.kwargs.get("md_regularization", 1e-5))),
            covariance_type=str(self.md_best_state.get("covariance_type", self.kwargs.get("md_covariance_type", "full"))),
            domain_strategy=str(self.md_best_state.get("domain_strategy", self.kwargs.get("md_domain_strategy", "source_target_min"))),
            train_metadata=train_bank.metadata,
            test_metadata=test_bank.metadata,
            test=True,
        )

        mean_score = float(np.mean(anomaly_scores))
        num_samples = float(len(test_bank))

        self.log("num_test_samples", num_samples, add_dataloader_idx=False, sync_dist=True)
        self.log("mean_anomaly_score", mean_score, add_dataloader_idx=False, sync_dist=True, prog_bar=True)

        if self.trainer.is_global_zero:
            self._save_test_outputs(test_bank.metadata.copy(), anomaly_scores)

        return {"num_test_samples": num_samples, "mean_anomaly_score": mean_score}

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    def _save_test_outputs(self, metadata: pd.DataFrame, anomaly_scores: np.ndarray) -> None:
        metadata = metadata.reset_index(drop=True).copy()
        metadata["anomaly_score"] = np.asarray(anomaly_scores, dtype=np.float32)

        basename_col = metadata["audio_path"].astype(str).apply(lambda x: Path(x).name)
        sub_eval = pd.DataFrame({
            "filename": basename_col,
            "anomaly_score": metadata["anomaly_score"],
            "machine": metadata["machine"].astype(str),
        })

        evaluator_root = self.kwargs.get(
            "evaluator_output_root",
            "/home/user/PSC/work/ASD/2026/dcase2025_task2_evaluator/teams",
        )
        save_dir = Path(evaluator_root) / str(self.kwargs.get("exp", "sep_exp"))
        save_dir.mkdir(parents=True, exist_ok=True)

        for machine in sub_eval["machine"].unique():
            temp = sub_eval.loc[sub_eval["machine"] == machine, ["filename", "anomaly_score"]].copy()
            temp.to_csv(save_dir / f"anomaly_score_{machine}_section_00_test.csv", index=False, header=False)
            temp.to_csv(save_dir / f"decision_result_{machine}_section_00_test.csv", index=False, header=False)

    def update_best_checkpoints(self, current_score: float) -> None:
        checkpoint_dir = Path("exp1") / str(self.kwargs.get("exp", "sep_exp")) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{self.current_epoch}_score{current_score:.4f}.pth"
        ckpt = {
            "state_dict": self.state_dict(),
            "epoch": int(self.current_epoch),
            "final_score": float(current_score),
            "md_state": dict(self.md_best_state),
            "hparams": dict(self.kwargs),
        }
        torch.save(ckpt, checkpoint_path)

        self.best_checkpoints.append((float(current_score), str(checkpoint_path)))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

        if len(self.best_checkpoints) > self.save_top_k:
            _, worst_path = self.best_checkpoints.pop(-1)
            worst_path = str(worst_path)
            if os.path.exists(worst_path):
                os.remove(worst_path)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        num_samples = len(self.train_dataloader().dataset)
        optimizer, scheduler = create_optimizer_scheduler(self.parameters(), num_samples, **self.kwargs)
        return [optimizer], [scheduler]


__all__ = ["ssmodule_sep"]
