#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Source-separation proxy evaluation / validation launcher
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/eval_sep.sh validate
#   bash scripts/eval_sep.sh test
#   MODE=test EXP_NAME=my_sep_exp GPU=1 TEST_CSV=/path/eval.csv bash scripts/eval_sep.sh
#
# Positional mode is optional. You can also set MODE=validate or MODE=test.
# -----------------------------------------------------------------------------

MODE="${1:-${MODE:-validate}}"
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

ENABLE_LOGGING="${ENABLE_LOGGING:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-2026_sep_eval}"

if [[ "${MODE}" != "validate" && "${MODE}" != "test" ]]; then
  echo "Invalid mode: ${MODE}"
  echo "Use: validate or test"
  exit 1
fi

CMD=(
  python train_sep.py
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
)

if [[ -n "${EXTRACT_SEGMENT_SECONDS}" ]]; then
  CMD+=(--extract_segment_seconds "${EXTRACT_SEGMENT_SECONDS}")
fi

if [[ "${MODE}" == "validate" ]]; then
  CMD+=(--validate)
else
  CMD+=(--test)
fi

if [[ "${ENABLE_LOGGING}" == "true" ]]; then
  CMD+=(--logging --wandb_project "${WANDB_PROJECT}")
else
  CMD+=(--no-logging)
fi

printf 'Command: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
