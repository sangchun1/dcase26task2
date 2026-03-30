import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")

import os
import glob
import argparse
from typing import Optional, Union, List, Mapping, Any

import torch
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from ssmodule_sep import ssmodule_sep


# -----------------------------------------------------------------------------
# Trainer wrappers
# -----------------------------------------------------------------------------
def train(
    model: ssmodule_sep,
    logger: Optional[WandbLogger],
    args: dict,
) -> ssmodule_sep:
    lr_callback = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[int(d) for d in str(args["devices"]).split(",") if d.strip()] if torch.cuda.is_available() else 1,
        logger=logger,
        callbacks=[lr_callback],
        max_epochs=args["epochs"],
        accumulate_grad_batches=args["accumulation_steps"],
        precision=args.get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        num_sanity_val_steps=0,
        fast_dev_run=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    return model


def validate(
    model: ssmodule_sep,
    logger: Optional[WandbLogger],
    args: dict,
):
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[int(d) for d in str(args["devices"]).split(",") if d.strip()] if torch.cuda.is_available() else 1,
        logger=logger,
        precision=args.get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        num_sanity_val_steps=0,
        fast_dev_run=False,
    )
    return trainer.validate(model)


def test(
    model: ssmodule_sep,
    args: dict,
):
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[int(d) for d in str(args["devices"]).split(",") if d.strip()] if torch.cuda.is_available() else 1,
        callbacks=None,
        precision=args.get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
        fast_dev_run=False,
    )
    return trainer.test(model)


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def load_best_model(exp_name: str) -> str:
    checkpoint_dir = f"exp1/{exp_name}/checkpoints"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch*_score*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    best_checkpoint = None
    best_score = float("-inf")
    for ckpt_file in checkpoint_files:
        basename = os.path.basename(ckpt_file)
        try:
            score_str = basename.split("score")[-1].replace(".pth", "")
            score = float(score_str)
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_checkpoint = ckpt_file

    if best_checkpoint is None:
        raise FileNotFoundError(f"Unable to parse checkpoint score in {checkpoint_dir}")

    print(f"Loading best model from {best_checkpoint} with final score {best_score:.4f}")
    return best_checkpoint


def load_state_dict_from_checkpoint(model: ssmodule_sep, ckpt_path: str, device: str) -> ssmodule_sep:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded.")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    return model


# -----------------------------------------------------------------------------
# Argument definition
# -----------------------------------------------------------------------------
def define_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # General / runtime
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated GPU device IDs")
    parser.add_argument("--word_size", type=int, default=1, help="Number of GPUs for LR scaling")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--logging", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--precision", type=str, default="16-mixed")

    # Paths
    parser.add_argument("--train_path", type=str, default="/home/user/PSC/ASD/2026/data/pretrain_6.csv")
    parser.add_argument("--val_path", type=str, default="/home/user/PSC/ASD/2026/data/dev_test.csv")
    parser.add_argument("--test_path", type=str, default="/mnt/storage1/asd/2025_eval/eval.csv")
    parser.add_argument("--dev_train_path", type=str, default="/home/user/PSC/ASD/2026/data/dev_train.csv")
    parser.add_argument("--noise_csv_path", type=str, default=None)
    parser.add_argument("--finetune_from", type=str, default=None, help="Experiment name to load before training")

    # Training
    parser.add_argument("--seed", type=int, default=21208)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--extract_batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr_decay_rate", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine_restart", help="linear, cosine, cosine_restart")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--restart_period", type=int, default=5)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--pin_memory", default=True, action=argparse.BooleanOptionalAction)

    # Separation dataset / mixing
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--snr_min_db", type=float, default=-5.0)
    parser.add_argument("--snr_max_db", type=float, default=5.0)
    parser.add_argument("--train_fixed_snr_db", type=float, default=None)
    parser.add_argument(
        "--interference_mode",
        type=str,
        default="other_machine",
        choices=["other_machine", "same_machine", "any_machine", "external_noise", "other_machine_or_noise"],
    )
    parser.add_argument("--filter_normal_only", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--return_realized_snr", default=True, action=argparse.BooleanOptionalAction)

    # Raw extraction dataset
    parser.add_argument("--extract_segment_seconds", type=float, default=None)
    parser.add_argument("--extract_segment_mode", type=str, default="center", choices=["center", "random"])
    parser.add_argument("--extract_zero_mean", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--extract_peak_normalize", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--feature_bank_l2_normalize", default=False, action=argparse.BooleanOptionalAction)

    # Frontend
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--window", type=str, default="hann")
    parser.add_argument("--stft_center", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--stft_normalized", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input_representation", type=str, default="magnitude")
    parser.add_argument("--target_representation", type=str, default="magnitude")
    parser.add_argument("--mag_eps", type=float, default=1e-8)

    # Separator backbone
    parser.add_argument("--sep_in_channels", type=int, default=1)
    parser.add_argument("--sep_encoder_channels", type=str, default="32,64,128")
    parser.add_argument("--sep_bottleneck_channels", type=int, default=None)
    parser.add_argument("--sep_stem_channels", type=int, default=None)
    parser.add_argument("--sep_num_encoder_res_blocks", type=int, default=2)
    parser.add_argument("--sep_num_bottleneck_res_blocks", type=int, default=2)
    parser.add_argument("--sep_num_decoder_res_blocks", type=int, default=2)
    parser.add_argument("--sep_downsample_mode", type=str, default="conv")
    parser.add_argument("--sep_upsample_mode", type=str, default="bilinear")
    parser.add_argument("--sep_output_mode", type=str, default="mask", choices=["mask", "direct"])
    parser.add_argument("--sep_output_activation", type=str, default=None)
    parser.add_argument("--sep_return_all_encoder_features", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sep_return_decoder_features", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sep_asd_feature_source", type=str, default="encoder_bottleneck")

    # Feature head
    parser.add_argument("--feature_pooling", type=str, default="mean")
    parser.add_argument("--feature_projection_dim", type=int, default=None)
    parser.add_argument("--feature_aggregation", type=str, default="concat")
    parser.add_argument("--feature_dropout", type=float, default=0.0)
    parser.add_argument("--feature_use_layernorm", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--feature_l2_normalize", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--feature_detach_maps", default=False, action=argparse.BooleanOptionalAction)

    # Separation loss
    parser.add_argument("--l1_spec_weight", type=float, default=1.0)
    parser.add_argument("--mse_spec_weight", type=float, default=0.0)
    parser.add_argument("--spectral_convergence_weight", type=float, default=0.0)
    parser.add_argument("--sisdr_weight", type=float, default=0.1)
    parser.add_argument("--mask_l1_weight", type=float, default=0.0)
    parser.add_argument("--mask_bce_weight", type=float, default=0.0)
    parser.add_argument("--zero_mean_waveform", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sep_loss_reduction", type=str, default="mean")

    # Validation separation metrics
    parser.add_argument("--log_train_sep_metrics", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_sisdr", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_sisdri", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_snr", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_spec_l1", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_spec_mse", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_spectral_convergence", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_compute_mask_l1", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_zero_mean_sisdr", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_detach", default=True, action=argparse.BooleanOptionalAction)

    # Mahalanobis scoring
    parser.add_argument("--md_regularization", type=float, default=1e-5)
    parser.add_argument("--md_covariance_type", type=str, default="full", choices=["full", "diag"])
    parser.add_argument(
        "--md_domain_strategy",
        type=str,
        default="source_target_min",
        choices=["source_target_min", "source_target_mean", "source_only", "target_only", "global"],
    )
    parser.add_argument("--max_fpr", type=float, default=0.1)

    # Execution flags
    parser.add_argument("--train", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--validate", default=False, action=argparse.BooleanOptionalAction)

    # Logging / experiment
    parser.add_argument("--exp", type=str, default="sep_exp")
    parser.add_argument("--recipe", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--code_test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--evaluator_output_root",
        type=str,
        default="/home/user/KMJ/work/ASD/2026/dcase2025_task2_evaluator/teams",
    )
    return parser


def _parse_encoder_channels(text_or_value: Any):
    if isinstance(text_or_value, (tuple, list)):
        return tuple(int(v) for v in text_or_value)
    if text_or_value is None:
        return (32, 64, 128)
    text = str(text_or_value).strip()
    if not text:
        return (32, 64, 128)
    return tuple(int(tok.strip()) for tok in text.split(",") if tok.strip())



def get_args() -> dict:
    parser = argparse.ArgumentParser(description="Argument parser for separation-proxy ASD training.")
    parser = define_args(parser)
    args = parser.parse_args()
    out = vars(args)
    out["sep_encoder_channels"] = _parse_encoder_channels(out["sep_encoder_channels"])
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = get_args()

    if args["seed"] > 0:
        seed_everything(args["seed"], workers=True)
    else:
        print("Not seeding experiment.")

    if args["wandb_project"] is not None:
        project_name = args["wandb_project"]
    elif args["recipe"]:
        project_name = "2026_sep_recipe"
    elif args["code_test"]:
        project_name = "2026_sep_test"
    else:
        project_name = "2026_sep_train"

    logger: Optional[WandbLogger]
    if args["logging"]:
        logger = WandbLogger(project=project_name, name=args["exp"])
    else:
        logger = None

    model = ssmodule_sep(**args)

    device = f"cuda:{str(args['devices']).split(',')[0].strip()}" if torch.cuda.is_available() else "cpu"

    if args["train"]:
        if args["finetune_from"] is not None:
            ckpt_path = load_best_model(args["finetune_from"])
            model = load_state_dict_from_checkpoint(model, ckpt_path, device=device)
        model = train(model, logger, args)

    if args["validate"]:
        ckpt_path = load_best_model(args["exp"])
        model = load_state_dict_from_checkpoint(model, ckpt_path, device=device)
        results = validate(model, logger, args)
        print(results)

    if args["test"]:
        ckpt_path = load_best_model(args["exp"])
        model = load_state_dict_from_checkpoint(model, ckpt_path, device=device)
        results = test(model, args)
        print(results)
