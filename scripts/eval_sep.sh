#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Source-separation proxy evaluation / validation launcher
# Extended for external separator pretraining and 2nd-place-team style modules.
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/eval_sep.sh validate
#   bash scripts/eval_sep.sh test
#   MODE=test PRETRAINED_SEP_CKPT=/path/to/audiosep.ckpt bash scripts/eval_sep.sh
#
# If you have not yet renamed train_sep_modified.py back to train_sep.py,
# set TRAIN_SCRIPT explicitly:
#   TRAIN_SCRIPT=train_sep_modified.py bash scripts/eval_sep.sh validate
# -----------------------------------------------------------------------------

MODE="${1:-${MODE:-validate}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_sep.py}"

EXP_NAME="${EXP_NAME:-sep_resunet_v1}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
PRECISION="${PRECISION:-16-mixed}"

DEV_TRAIN_CSV="${DEV_TRAIN_CSV:-/home/user/PSC/ASD/2026/data/dev_train.csv}"
VAL_CSV="${VAL_CSV:-/home/user/PSC/ASD/2026/data/dev_test.csv}"
TEST_CSV="${TEST_CSV:-/mnt/storage1/asd/2025_eval/eval.csv}"

SAMPLE_RATE="${SAMPLE_RATE:-16000}"
EXTRACT_SEGMENT_SECONDS="${EXTRACT_SEGMENT_SECONDS:-}"
EXTRACT_SEGMENT_MODE="${EXTRACT_SEGMENT_MODE:-center}"
FEATURE_POOLING="${FEATURE_POOLING:-mean}"
FEATURE_AGGREGATION="${FEATURE_AGGREGATION:-concat}"
SEP_ENCODER_CHANNELS="${SEP_ENCODER_CHANNELS:-32,64,128}"
SEP_OUTPUT_MODE="${SEP_OUTPUT_MODE:-mask}"
SEP_UPSAMPLE_MODE="${SEP_UPSAMPLE_MODE:-bilinear}"
MD_REGULARIZATION="${MD_REGULARIZATION:-1e-5}"
MD_COVARIANCE_TYPE="${MD_COVARIANCE_TYPE:-full}"
MD_DOMAIN_STRATEGY="${MD_DOMAIN_STRATEGY:-source_target_min}"

TEAM_NAME="${TEAM_NAME:-default_team}"
EVALUATOR_OUTPUT_ROOT="${EVALUATOR_OUTPUT_ROOT:-/home/user/PSC/ASD/2026/dcase2025_task2_evaluator/teams}"
DECISION_PERCENTILE="${DECISION_PERCENTILE:-95}"

ENABLE_LOGGING="${ENABLE_LOGGING:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-2026_sep_eval}"

# External pretrained initialization
PRETRAINED_SEP_CKPT=${PRETRAINED_SEP_CKPT:-/home/user/PSC/ASD/2026/checkpoints/audiosep_base_4M_steps.ckpt}
PRETRAINED_SEP_STRICT_BACKBONE="${PRETRAINED_SEP_STRICT_BACKBONE:-false}"
PRETRAINED_GUIDE_CKPT="${PRETRAINED_GUIDE_CKPT:-}"
PRETRAINED_GUIDE_STRICT="${PRETRAINED_GUIDE_STRICT:-false}"
LOAD_EXTERNAL_PRETRAINED_IN_DRIVER="${LOAD_EXTERNAL_PRETRAINED_IN_DRIVER:-true}"

# Optional model extensions
GUIDE_CLASS_MODE="${GUIDE_CLASS_MODE:-machine}"
USE_STAGE2_SED="${USE_STAGE2_SED:-false}"
GUIDE_NUM_CLASSES="${GUIDE_NUM_CLASSES:-1}"
GUIDE_DEFAULT_CLASS_INDEX="${GUIDE_DEFAULT_CLASS_INDEX:-0}"
USE_TIME_FILM="${USE_TIME_FILM:-false}"
USE_LATENT_INJECTION="${USE_LATENT_INJECTION:-false}"
USE_DPRNN="${USE_DPRNN:-false}"
USE_ITERATIVE_REFINEMENT="${USE_ITERATIVE_REFINEMENT:-false}"
REFINEMENT_NUM_ITERATIONS="${REFINEMENT_NUM_ITERATIONS:-2}"

if [[ "${MODE}" != "validate" && "${MODE}" != "test" ]]; then
  echo "Invalid mode: ${MODE}"
  echo "Use: validate or test"
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --exp "${EXP_NAME}"
  --devices "${GPU}"
  --num_workers "${NUM_WORKERS}"
  --extract_batch_size "${EXTRACT_BATCH_SIZE}"
  --precision "${PRECISION}"
  --dev_train_path "${DEV_TRAIN_CSV}"
  --val_path "${VAL_CSV}"
  --test_path "${TEST_CSV}"
  --sample_rate "${SAMPLE_RATE}"
  --extract_segment_mode "${EXTRACT_SEGMENT_MODE}"
  --feature_pooling "${FEATURE_POOLING}"
  --feature_aggregation "${FEATURE_AGGREGATION}"
  --sep_encoder_channels "${SEP_ENCODER_CHANNELS}"
  --sep_output_mode "${SEP_OUTPUT_MODE}"
  --sep_upsample_mode "${SEP_UPSAMPLE_MODE}"
  --md_regularization "${MD_REGULARIZATION}"
  --md_covariance_type "${MD_COVARIANCE_TYPE}"
  --md_domain_strategy "${MD_DOMAIN_STRATEGY}"
  --team_name "${TEAM_NAME}"
  --evaluator_output_root "${EVALUATOR_OUTPUT_ROOT}"
  --decision_percentile "${DECISION_PERCENTILE}"
  --guide_class_mode "${GUIDE_CLASS_MODE}"
  --guide_num_classes "${GUIDE_NUM_CLASSES}"
  --guide_default_class_index "${GUIDE_DEFAULT_CLASS_INDEX}"
  --refinement_num_iterations "${REFINEMENT_NUM_ITERATIONS}"
)

if [[ -n "${EXTRACT_SEGMENT_SECONDS}" ]]; then
  CMD+=(--extract_segment_seconds "${EXTRACT_SEGMENT_SECONDS}")
fi

if [[ "${MODE}" == "validate" ]]; then
  CMD+=(--validate)
else
  CMD+=(--test)
fi

if [[ -n "${PRETRAINED_SEP_CKPT}" ]]; then
  CMD+=(--pretrained_sep_ckpt "${PRETRAINED_SEP_CKPT}")
fi
if [[ "${PRETRAINED_SEP_STRICT_BACKBONE}" == "true" ]]; then
  CMD+=(--pretrained_sep_strict_backbone)
fi
if [[ -n "${PRETRAINED_GUIDE_CKPT}" ]]; then
  CMD+=(--pretrained_guide_ckpt "${PRETRAINED_GUIDE_CKPT}")
fi
if [[ "${PRETRAINED_GUIDE_STRICT}" == "true" ]]; then
  CMD+=(--pretrained_guide_strict)
fi
if [[ "${LOAD_EXTERNAL_PRETRAINED_IN_DRIVER}" == "false" ]]; then
  CMD+=(--no-load_external_pretrained_in_driver)
fi

if [[ "${USE_STAGE2_SED}" == "true" ]]; then
  CMD+=(--use_stage2_sed)
fi
if [[ "${USE_TIME_FILM}" == "true" ]]; then
  CMD+=(--use_time_film)
fi
if [[ "${USE_LATENT_INJECTION}" == "true" ]]; then
  CMD+=(--use_latent_injection)
fi
if [[ "${USE_DPRNN}" == "true" ]]; then
  CMD+=(--use_dprnn)
fi
if [[ "${USE_ITERATIVE_REFINEMENT}" == "true" ]]; then
  CMD+=(--use_iterative_refinement)
fi

if [[ "${ENABLE_LOGGING}" == "true" ]]; then
  CMD+=(--logging --wandb_project "${WANDB_PROJECT}")
else
  CMD+=(--no-logging)
fi

printf 'Command: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
