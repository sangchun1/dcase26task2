"""Lightning module for source-separation proxy training and ASD scoring.

This version extends the original separation-based module with hooks for the
DCASE 2025 Task 4 2nd-place team's ideas:

1. optional stage-2 SED / guide encoder
2. optional Time-FiLM conditioning and hidden-state injection
3. optional iterative refinement wrapper
4. optional external pretrained separator loading (e.g. AudioSep-style ckpt)

It is written to stay backward-compatible with the original simplified setup:

    synthetic mixture -> separator training -> separator feature extraction
    -> Mahalanobis anomaly scoring
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
from model.separator.stage2_sed import Stage2SEDConfig, Stage2SEDGuideEncoder
from model.separator.conditioning import gather_class_probability_map
from model.separator.feature_head import SeparatorFeatureHead, FeatureHeadConfig
from model.separator.iterative_refinement import (
    IterativeRefinementConfig,
    IterativeRefinementWrapper,
)
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

    Additional high-level kwargs in this extended version
    -----------------------------------------------------
    Pretrained / transfer:
    - pretrained_sep_ckpt
    - pretrained_sep_strict_backbone
    - pretrained_sep_strip_prefixes

    Guide encoder / conditioning:
    - use_stage2_sed
    - guide_num_classes
    - guide_default_class_index
    - use_time_film
    - use_latent_injection

    Iterative refinement:
    - use_iterative_refinement
    - refinement_num_iterations
    - refinement_detach_between_iterations
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = dict(kwargs)
        self.save_hyperparameters(ignore=[])

        # ------------------------------------------------------------------
        # Core components
        # ------------------------------------------------------------------
        self.frontend = self._build_frontend()
        self.guide_encoder = self._build_guide_encoder()
        self.separator = self._build_separator()
        self.feature_head = self._build_feature_head()
        self.loss_fn = self._build_loss()
        self.metric_cfg = self._build_metric_config()
        self.metric_tracker = SeparationMetricTracker()

        self.pretrained_load_info: Dict[str, Any] = {}
        self._maybe_load_pretrained_separator()
        self._maybe_load_pretrained_guide_encoder()

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
    
    def _norm_guide_token(self, value) -> str:
        if value is None:
            return "__UNK__"
        text = str(value).strip()
        return text if text else "__UNK__"

    def _make_guide_token_from_series(self, row: pd.Series) -> str:
        mode = str(self.kwargs.get("guide_class_mode", "machine"))

        machine = self._norm_guide_token(row.get("machine", None))
        domain = self._norm_guide_token(row.get("domain", None))
        attribute = self._norm_guide_token(row.get("attribute", None))
        year = self._norm_guide_token(row.get("year", None))

        if mode == "machine":
            return machine
        if mode == "machine_domain":
            return f"{machine}::{domain}"
        if mode == "machine_attribute":
            return f"{machine}::{attribute}"
        if mode == "machine_year":
            return f"{machine}::{year}"
        if mode == "domain":
            return domain
        if mode == "attribute":
            return attribute

        # fallback
        return machine

    def _infer_guide_vocab_from_csv(self) -> Dict[str, int]:
        csv_candidates = [
            self.kwargs.get("train_path", None),
            self.kwargs.get("dev_train_path", None),
            self.kwargs.get("val_path", None),
        ]
        csv_path = None
        for path in csv_candidates:
            if path and os.path.exists(path):
                csv_path = path
                break

        if csv_path is None:
            return {"__DEFAULT__": 0}

        df = pd.read_csv(csv_path)

        if bool(self.kwargs.get("filter_normal_only", True)) and "label" in df.columns:
            labels = df["label"].astype(str).str.lower()
            df = df[labels != "anomaly"]

        tokens = sorted({self._make_guide_token_from_series(row) for _, row in df.iterrows()})
        if len(tokens) == 0:
            tokens = ["__DEFAULT__"]

        return {token: idx for idx, token in enumerate(tokens)}

    def _get_or_build_guide_vocab(self) -> Dict[str, int]:
        vocab = getattr(self, "_guide_vocab", None)
        if vocab is None:
            vocab = self._infer_guide_vocab_from_csv()
            self._guide_vocab = vocab
        return vocab

    def _resolve_guide_num_classes(self) -> int:
        configured = self.kwargs.get("guide_num_classes", 0)
        if configured is not None and int(configured) > 0:
            return int(configured)
        vocab = self._get_or_build_guide_vocab()
        return max(len(vocab), 1)

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

    def _use_guide_encoder(self) -> bool:
        return bool(
            self.kwargs.get("use_stage2_sed", False)
            or self.kwargs.get("use_time_film", False)
            or self.kwargs.get("use_latent_injection", False)
        )

    def _build_guide_encoder(self) -> Optional[Stage2SEDGuideEncoder]:
        if not self._use_guide_encoder():
            self.guide_num_classes = 0
            return None

        self.guide_num_classes = self._resolve_guide_num_classes()
        self.kwargs["guide_num_classes"] = self.guide_num_classes

        guide_cfg = Stage2SEDConfig(
            num_classes=self.guide_num_classes,
            input_channels=int(self.kwargs.get("guide_input_channels", 1)),
            stem_channels=int(self.kwargs.get("guide_stem_channels", 64)),
            hidden_dim=int(self.kwargs.get("guide_hidden_dim") or 256),
            num_layers=int(self.kwargs.get("guide_num_layers", 4)),
            num_heads=int(self.kwargs.get("guide_num_heads", 8)),
            mlp_ratio=float(self.kwargs.get("guide_mlp_ratio", 4.0)),
            dropout=float(self.kwargs.get("guide_dropout", 0.1)),
            max_time_positions=int(self.kwargs.get("guide_max_time_positions", 4096)),
            use_frequency_attention_pool=bool(self.kwargs.get("guide_use_frequency_attention_pool", True)),
            temporal_conv_kernel_size=int(self.kwargs.get("guide_temporal_conv_kernel_size", 5)),
            return_all_hidden_states=bool(self.kwargs.get("guide_return_all_hidden_states", True)),
            strong_activation=str(self.kwargs.get("guide_strong_activation", "sigmoid")),
        )
        return Stage2SEDGuideEncoder(guide_cfg)
    
    def _convert_single_class_value_to_index(self, value) -> int:
        default_idx = int(self.kwargs.get("guide_default_class_index", 0))
        vocab = self._get_or_build_guide_vocab()

        if value is None:
            return default_idx

        if isinstance(value, (int, np.integer)):
            return int(value)

        token = self._norm_guide_token(value)
        return int(vocab.get(token, default_idx))

    def _extract_safe_guide_class_index(self, batch: Dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        candidate = None
        for key in ("guide_class_index", "target_class_index", "class_index", "target_machine_index", "machine_index", "target_machine"):
            if key in batch:
                candidate = batch[key]
                break

        if candidate is None:
            idx = torch.full(
                (batch_size,),
                int(self.kwargs.get("guide_default_class_index", 0)),
                dtype=torch.long,
                device=device,
            )
        elif isinstance(candidate, torch.Tensor):
            idx = candidate.to(device=device, dtype=torch.long).view(-1)
        elif isinstance(candidate, (list, tuple)):
            idx = torch.tensor(
                [self._convert_single_class_value_to_index(v) for v in candidate],
                dtype=torch.long,
                device=device,
            )
        else:
            idx = torch.full(
                (batch_size,),
                self._convert_single_class_value_to_index(candidate),
                dtype=torch.long,
                device=device,
            )

        num_classes = max(int(getattr(self, "guide_num_classes", 0) or self.kwargs.get("guide_num_classes", 1)), 1)
        idx = idx.clamp(min=0, max=num_classes - 1)
        return idx

    def _build_separator(self) -> torch.nn.Module:
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
            use_dprnn=bool(self.kwargs.get("use_dprnn", False)),
            dprnn_hidden_size=int(self.kwargs.get("dprnn_hidden_size") or 256),
            dprnn_num_layers=int(self.kwargs.get("dprnn_num_layers") or 2),
            dprnn_dropout=float(self.kwargs.get("dprnn_dropout", 0.0)),
            dprnn_bidirectional=bool(self.kwargs.get("dprnn_bidirectional", True)),
            dprnn_rnn_type=str(self.kwargs.get("dprnn_rnn_type", "gru")),
            use_time_film=bool(self.kwargs.get("use_time_film", False)),
            time_film_on_bottleneck=bool(self.kwargs.get("time_film_on_bottleneck", True)),
            time_film_on_decoder=bool(self.kwargs.get("time_film_on_decoder", False)),
            time_film_condition_dim=int(self.kwargs.get("time_film_condition_dim") or 1),
            time_film_hidden_dim=int(self.kwargs.get("time_film_hidden_dim") or 128),
            time_film_num_layers=int(self.kwargs.get("time_film_num_layers") or 2),
            time_film_dropout=float(self.kwargs.get("time_film_dropout", 0.0)),
            time_film_residual_gamma=bool(self.kwargs.get("time_film_residual_gamma", True)),
            use_latent_injection=bool(self.kwargs.get("use_latent_injection", False)),
            latent_injection_input_dim=int(self.kwargs.get("latent_injection_input_dim") or 256),
            latent_injection_hidden_dim=int(self.kwargs.get("latent_injection_hidden_dim") or 256),
            latent_injection_num_hidden_states=int(self.kwargs.get("latent_injection_num_hidden_states") or 4),
        )
        separator: torch.nn.Module = ResUNetSeparator(sep_cfg)

        if bool(self.kwargs.get("use_iterative_refinement", False)):
            refinement_cfg = IterativeRefinementConfig(
                enabled=True,
                num_iterations=int(self.kwargs.get("refinement_num_iterations", 2)),
                detach_between_iterations=bool(self.kwargs.get("refinement_detach_between_iterations", True)),
                base_input_channels=int(self.kwargs.get("sep_in_channels", 1)),
                refinement_channels=int(self.kwargs.get("refinement_channels") or 1),
                refinement_signal_key=str(self.kwargs.get("refinement_signal_key", "pred_spec")),
                adapter_hidden_channels=int(self.kwargs.get("refinement_adapter_hidden_channels") or 16),
                adapter_num_layers=int(self.kwargs.get("refinement_adapter_num_layers") or 2),
                adapter_activation=str(self.kwargs.get("refinement_adapter_activation", "silu")),
                residual_to_base_input=bool(self.kwargs.get("refinement_residual_to_base_input", True)),
                return_history=bool(self.kwargs.get("refinement_return_history", False)),
            )
            separator = IterativeRefinementWrapper(separator=separator, config=refinement_cfg)

        return separator

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
    # Pretrained loading helpers
    # ------------------------------------------------------------------
    def _unwrap_base_separator(self) -> ResUNetSeparator:
        separator = self.separator
        if isinstance(separator, IterativeRefinementWrapper):
            separator = separator.separator
        if not isinstance(separator, ResUNetSeparator):
            raise TypeError(f"Expected base separator to be ResUNetSeparator, got {type(separator)!r}")
        return separator

    @staticmethod
    def _load_checkpoint_payload(checkpoint_path: Union[str, os.PathLike]) -> Mapping[str, Any]:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(ckpt, Mapping):
            return ckpt
        raise TypeError(f"Checkpoint must load to a mapping, got {type(ckpt)!r}")

    @staticmethod
    def _extract_state_dict_from_checkpoint(ckpt: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        for key in ("state_dict", "model_state_dict", "model", "separator_state_dict"):
            value = ckpt.get(key, None)
            if isinstance(value, Mapping):
                return value
        return ckpt  # raw state_dict case

    def _maybe_load_pretrained_separator(self) -> None:
        ckpt_path = self.kwargs.get("pretrained_sep_ckpt", None)
        if not ckpt_path:
            return

        ckpt = self._load_checkpoint_payload(ckpt_path)
        state_dict = self._extract_state_dict_from_checkpoint(ckpt)
        base_separator = self._unwrap_base_separator()
        load_info = base_separator.load_pretrained_separator_state_dict(
            state_dict=state_dict,
            strict_backbone=bool(self.kwargs.get("pretrained_sep_strict_backbone", False)),
            strip_prefixes=tuple(
                self.kwargs.get(
                    "pretrained_sep_strip_prefixes",
                    ("separator.", "model.separator.", "module.separator.", "module."),
                )
            ),
        )
        load_info["checkpoint_path"] = str(ckpt_path)
        self.pretrained_load_info["separator"] = load_info

    def _maybe_load_pretrained_guide_encoder(self) -> None:
        if self.guide_encoder is None:
            return
        ckpt_path = self.kwargs.get("pretrained_guide_ckpt", None)
        if not ckpt_path:
            return

        ckpt = self._load_checkpoint_payload(ckpt_path)
        state_dict = self._extract_state_dict_from_checkpoint(ckpt)
        incompatible = self.guide_encoder.load_state_dict(state_dict, strict=bool(self.kwargs.get("pretrained_guide_strict", False)))
        self.pretrained_load_info["guide_encoder"] = {
            "checkpoint_path": str(ckpt_path),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

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

    def _resolve_batch_class_index(self, batch: Optional[Mapping[str, Any]], batch_size: int, device: torch.device) -> Optional[Union[int, torch.Tensor]]:
        if not self._use_guide_encoder():
            return None

        candidate_keys = (
            "target_class_index",
            "class_index",
            "guide_class_index",
            "target_machine_index",
            "machine_index",
        )
        if batch is not None:
            for key in candidate_keys:
                if key not in batch:
                    continue
                value = batch[key]
                if torch.is_tensor(value):
                    value = value.to(device=device)
                    if value.ndim == 0:
                        return int(value.item())
                    if value.ndim == 1 and value.shape[0] == batch_size:
                        return value.long()
                elif isinstance(value, (int, np.integer)):
                    return int(value)
                elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                    return torch.as_tensor(value, dtype=torch.long, device=device)

        default_index = int(self.kwargs.get("guide_default_class_index", 0))
        num_classes = int(self.kwargs.get("guide_num_classes", 1))
        if num_classes == 1:
            return 0
        return default_index

    def _compute_separator_guidance(
        self,
        mix_spec: torch.Tensor,
        batch: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.guide_encoder is None:
            return {
                "guide_out": None,
                "time_condition": None,
                "hidden_states": None,
                "class_index": None,
            }

        guide_out = self.guide_encoder(mix_spec)
        batch_size = input_spec.shape[0]
        class_index = self._extract_safe_guide_class_index(
            batch=batch,
            batch_size=batch_size,
            device=input_spec.device,
        )

        time_condition = gather_class_probability_map(strong_probabilities, class_index)

                hidden_states = None
                if bool(self.kwargs.get("use_latent_injection", False)):
                    hidden_states = guide_out.get("hidden_states", None)

                return {
                    "guide_out": guide_out,
                    "time_condition": time_condition,
                    "hidden_states": hidden_states,
                    "class_index": class_index,
                }

    def _run_separator(self, mix_wave: torch.Tensor, batch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        mix_spec, mix_aux = self.frontend.wave_to_input_spec(mix_wave)
        guidance = self._compute_separator_guidance(mix_spec=mix_spec, batch=batch)
        sep_out = self.separator(
            mix_spec,
            time_condition=guidance["time_condition"],
            hidden_states=guidance["hidden_states"],
        )
        if not isinstance(sep_out, dict):
            raise TypeError(f"separator must return a dict, got {type(sep_out)!r}")
        sep_out["mix_aux"] = mix_aux
        if guidance["guide_out"] is not None:
            sep_out["guide_out"] = guidance["guide_out"]
            sep_out["guide_time_condition"] = guidance["time_condition"]
            sep_out["guide_class_index"] = guidance["class_index"]
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

    def _extract_embedding_tensor(self, wave: torch.Tensor, batch: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        input_spec, _ = self.frontend.wave_to_input_spec(wave)
        guidance = self._compute_separator_guidance(mix_spec=input_spec, batch=batch)
        sep_out = self.separator(
            input_spec,
            time_condition=guidance["time_condition"],
            hidden_states=guidance["hidden_states"],
        )
        head_out = self.feature_head(sep_out)
        return head_out["embedding"]

    def _extract_embeddings_from_raw_batch(self, batch: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        metadata = self._strip_wave_from_metadata(batch)
        wave = batch["wave"]

        if torch.is_tensor(wave):
            wave = wave.to(self.device, non_blocking=True)
            emb = self._extract_embedding_tensor(wave, batch=batch)
            return emb, metadata

        embeddings: List[torch.Tensor] = []
        for item in wave:
            item = item.to(self.device, non_blocking=True)
            emb = self._extract_embedding_tensor(item.unsqueeze(0) if item.ndim == 2 else item, batch=batch)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0), metadata

    def _build_feature_bank_from_loader(self, loader: DataLoader, name: str) -> FeatureBank:
        accumulator = FeatureBankAccumulator(name=name)
        self.separator.eval()
        self.feature_head.eval()
        self.frontend.eval()
        if self.guide_encoder is not None:
            self.guide_encoder.eval()

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
        return self._run_separator(mix_wave, batch=None)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.pretrained_load_info:
            print("[pretrained_load_info]", self.pretrained_load_info)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        tensors = self._prepare_train_batch(batch)
        mix_wave = tensors["mix_wave"]
        target_wave = tensors["target_wave"]

        sep_out = self._run_separator(mix_wave, batch=batch)
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

        val_scores = compute_result_md(
            result_train=train_bank.embeddings,
            result_test=val_bank.embeddings,
            regularization=regularization,
            covariance_type=covariance_type,
            domain_strategy=domain_strategy,
            train_metadata=train_bank.metadata,
            test_metadata=val_bank.metadata,
            test=True,
        )
        decision_thresholds = self._compute_labeled_thresholds(
            metadata=val_bank.metadata,
            anomaly_scores=val_scores,
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
                "decision_thresholds": decision_thresholds,
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

        regularization = float(self.md_best_state.get("regularization", self.kwargs.get("md_regularization", 1e-5)))
        covariance_type = str(self.md_best_state.get("covariance_type", self.kwargs.get("md_covariance_type", "full")))
        domain_strategy = str(self.md_best_state.get("domain_strategy", self.kwargs.get("md_domain_strategy", "source_target_min")))

        anomaly_scores = compute_result_md(
            result_train=train_bank.embeddings,
            result_test=test_bank.embeddings,
            regularization=regularization,
            covariance_type=covariance_type,
            domain_strategy=domain_strategy,
            train_metadata=train_bank.metadata,
            test_metadata=test_bank.metadata,
            test=True,
        )

        decision_thresholds = self.md_best_state.get("decision_thresholds", None)
        if not decision_thresholds:
            train_scores = compute_result_md(
                result_train=train_bank.embeddings,
                result_test=train_bank.embeddings,
                regularization=regularization,
                covariance_type=covariance_type,
                domain_strategy=domain_strategy,
                train_metadata=train_bank.metadata,
                test_metadata=train_bank.metadata,
                test=True,
            )
            decision_thresholds = self._compute_percentile_thresholds(
                metadata=train_bank.metadata,
                anomaly_scores=train_scores,
                percentile=float(self.kwargs.get("decision_percentile", os.getenv("DECISION_PERCENTILE", 95.0))),
            )
            self.md_best_state["decision_thresholds"] = decision_thresholds

        mean_score = float(np.mean(anomaly_scores))
        num_samples = float(len(test_bank))

        self.log("num_test_samples", num_samples, add_dataloader_idx=False, sync_dist=True)
        self.log("mean_anomaly_score", mean_score, add_dataloader_idx=False, sync_dist=True, prog_bar=True)

        if self.trainer.is_global_zero:
            self._save_test_outputs(
                metadata=test_bank.metadata.copy(),
                anomaly_scores=anomaly_scores,
                decision_thresholds=decision_thresholds,
            )

        return {"num_test_samples": num_samples, "mean_anomaly_score": mean_score}

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _labels_to_binary(labels: pd.Series) -> np.ndarray:
        lowered = labels.astype(str).str.lower()
        mapped = lowered.map({"normal": 0, "anomaly": 1})
        if mapped.notna().all():
            return mapped.to_numpy(dtype=np.int64)
        return pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.int64)

    @staticmethod
    def _find_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if scores.size == 0:
            return 0.0

        candidates = np.unique(scores)
        candidates = np.concatenate([candidates, [float(candidates.max() + 1e-6)]])

        eps = 1e-12
        best_thr = float(candidates[0])
        best_f1 = -1.0

        for thr in candidates:
            pred = (scores >= thr).astype(np.int64)
            tp = int(np.sum((pred == 1) & (labels == 1)))
            fp = int(np.sum((pred == 1) & (labels == 0)))
            fn = int(np.sum((pred == 0) & (labels == 1)))

            prec = tp / max(tp + fp, eps)
            rec = tp / max(tp + fn, eps)
            f1 = 2.0 * prec * rec / max(prec + rec, eps)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        return best_thr

    def _compute_labeled_thresholds(
        self,
        metadata: pd.DataFrame,
        anomaly_scores: np.ndarray,
    ) -> Dict[str, Any]:
        df = metadata.reset_index(drop=True).copy()
        df["anomaly_score"] = np.asarray(anomaly_scores, dtype=np.float32)
        df["machine"] = df["machine"].astype(str)
        if "domain" in df.columns:
            df["domain"] = df["domain"].astype(str).str.lower()

        df["label_bin"] = self._labels_to_binary(df["label"])

        thresholds: Dict[str, Any] = {
            "global": self._find_best_f1_threshold(df["anomaly_score"].to_numpy(), df["label_bin"].to_numpy()),
            "per_machine": {},
            "per_machine_domain": {},
        }

        for machine, mdf in df.groupby("machine"):
            thresholds["per_machine"][machine] = self._find_best_f1_threshold(
                mdf["anomaly_score"].to_numpy(),
                mdf["label_bin"].to_numpy(),
            )

            if "domain" in mdf.columns:
                thresholds["per_machine_domain"][machine] = {}
                for domain, ddf in mdf.groupby("domain"):
                    thresholds["per_machine_domain"][machine][domain] = self._find_best_f1_threshold(
                        ddf["anomaly_score"].to_numpy(),
                        ddf["label_bin"].to_numpy(),
                    )

        return thresholds

    def _compute_percentile_thresholds(
        self,
        metadata: pd.DataFrame,
        anomaly_scores: np.ndarray,
        percentile: float,
    ) -> Dict[str, Any]:
        df = metadata.reset_index(drop=True).copy()
        df["anomaly_score"] = np.asarray(anomaly_scores, dtype=np.float32)
        df["machine"] = df["machine"].astype(str)
        if "domain" in df.columns:
            df["domain"] = df["domain"].astype(str).str.lower()

        thresholds: Dict[str, Any] = {
            "global": float(np.percentile(df["anomaly_score"].to_numpy(), percentile)),
            "per_machine": {},
            "per_machine_domain": {},
        }

        for machine, mdf in df.groupby("machine"):
            thresholds["per_machine"][machine] = float(
                np.percentile(mdf["anomaly_score"].to_numpy(), percentile)
            )

            if "domain" in mdf.columns:
                thresholds["per_machine_domain"][machine] = {}
                for domain, ddf in mdf.groupby("domain"):
                    thresholds["per_machine_domain"][machine][domain] = float(
                        np.percentile(ddf["anomaly_score"].to_numpy(), percentile)
                    )

        return thresholds

    @staticmethod
    def _resolve_decision_threshold(
        machine: str,
        domain: Optional[str],
        thresholds: Mapping[str, Any],
    ) -> float:
        if not isinstance(thresholds, Mapping):
            return 0.0

        per_machine_domain = thresholds.get("per_machine_domain", {})
        if domain is not None:
            domain = str(domain).lower()
            if machine in per_machine_domain and domain in per_machine_domain[machine]:
                return float(per_machine_domain[machine][domain])

        per_machine = thresholds.get("per_machine", {})
        if machine in per_machine:
            return float(per_machine[machine])

        if "global" in thresholds:
            return float(thresholds["global"])

        return 0.0

    def _resolve_evaluator_root(self) -> Path:
        legacy_default = "/home/user/PSC/ASD/2026/dcase2025_task2_evaluator/teams"
        repo_default = PROJECT_ROOT / "dcase2025_task2_evaluator" / "teams"
        modern_default = Path("/home/user/PSC/ASD/2026/dcase2025_task2_evaluator/teams")

        env_root = os.getenv("EVALUATOR_OUTPUT_ROOT")
        if env_root:
            return Path(env_root)

        kw_root = self.kwargs.get("evaluator_output_root", None)
        if kw_root:
            kw_root_path = Path(str(kw_root))
            if str(kw_root_path) != legacy_default:
                return kw_root_path

        if repo_default.exists():
            return repo_default

        if modern_default.exists():
            return modern_default

        if kw_root:
            return Path(str(kw_root))

        return modern_default

    def _resolve_team_name(self) -> str:
        return str(
            self.kwargs.get("team_name")
            or os.getenv("TEAM_NAME")
            or "default_team"
        )

    def _save_test_outputs(
        self,
        metadata: pd.DataFrame,
        anomaly_scores: np.ndarray,
        decision_thresholds: Mapping[str, Any],
    ) -> None:
        metadata = metadata.reset_index(drop=True).copy()
        metadata["anomaly_score"] = np.asarray(anomaly_scores, dtype=np.float32)

        basename_col = metadata["audio_path"].astype(str).apply(lambda x: Path(x).name)
        domain_col = (
            metadata["domain"].astype(str).str.lower()
            if "domain" in metadata.columns
            else pd.Series([""] * len(metadata), index=metadata.index)
        )
        sub_eval = pd.DataFrame({
            "filename": basename_col,
            "anomaly_score": metadata["anomaly_score"],
            "machine": metadata["machine"].astype(str),
            "domain": domain_col,
        })

        sub_eval["decision_result"] = [
            int(
                score >= self._resolve_decision_threshold(
                    machine=str(machine),
                    domain=(None if domain == "" else str(domain)),
                    thresholds=decision_thresholds,
                )
            )
            for machine, domain, score in zip(
                sub_eval["machine"],
                sub_eval["domain"],
                sub_eval["anomaly_score"],
            )
        ]

        evaluator_root = self._resolve_evaluator_root()
        team_name = self._resolve_team_name()
        system_name = str(self.kwargs.get("exp", "sep_exp"))

        save_dir = evaluator_root / team_name / system_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for machine in sub_eval["machine"].unique():
            temp_score = sub_eval.loc[
                sub_eval["machine"] == machine,
                ["filename", "anomaly_score"],
            ].copy()
            temp_decision = sub_eval.loc[
                sub_eval["machine"] == machine,
                ["filename", "decision_result"],
            ].copy()

            temp_score.to_csv(
                save_dir / f"anomaly_score_{machine}_section_00_test.csv",
                index=False,
                header=False,
            )
            temp_decision.to_csv(
                save_dir / f"decision_result_{machine}_section_00_test.csv",
                index=False,
                header=False,
            )

        print(f"Saved evaluator CSVs to: {save_dir}")

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
            "pretrained_load_info": dict(self.pretrained_load_info),
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
