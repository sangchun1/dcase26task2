#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Source-separation proxy training launcher (nohup-friendly)
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/train_sep_nohup.sh
#   EXP_NAME=my_sep_exp GPU=1 bash scripts/train_sep_nohup.sh
#   TRAIN_CSV=/path/train.csv DEV_TRAIN_CSV=/path/dev_train.csv VAL_CSV=/path/dev_test.csv \
#   bash scripts/train_sep_nohup.sh
#
# Notes:
# - This script starts training in the background with nohup.
# - Logs are written to: exp1/${EXP_NAME}/logs/train_nohup.log
# - Edit the paths below or override them with environment variables.
# -----------------------------------------------------------------------------

EXP_NAME="${EXP_NAME:-sep_resunet_v1}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-21208}"
PRECISION="${PRECISION:-16-mixed}"

TRAIN_CSV="${TRAIN_CSV:-/home/user/PSC/ASD/2026/data/pretrain_6.csv}"
DEV_TRAIN_CSV="${DEV_TRAIN_CSV:-/home/user/PSC/ASD/2026/data/dev_train.csv}"
VAL_CSV="${VAL_CSV:-/home/user/PSC/ASD/2026/data/dev_test.csv}"
NOISE_CSV="${NOISE_CSV:-}"

INTERFERENCE_MODE="${INTERFERENCE_MODE:-other_machine}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-2.0}"
SNR_MIN_DB="${SNR_MIN_DB:--5.0}"
SNR_MAX_DB="${SNR_MAX_DB:-5.0}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"

SEP_ENCODER_CHANNELS="${SEP_ENCODER_CHANNELS:-32,64,128}"
SEP_OUTPUT_MODE="${SEP_OUTPUT_MODE:-mask}"
SEP_UPSAMPLE_MODE="${SEP_UPSAMPLE_MODE:-bilinear}"
FEATURE_POOLING="${FEATURE_POOLING:-mean}"
FEATURE_AGGREGATION="${FEATURE_AGGREGATION:-concat}"

MAX_LR="${MAX_LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine_restart}"
RESTART_PERIOD="${RESTART_PERIOD:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"

L1_SPEC_WEIGHT="${L1_SPEC_WEIGHT:-1.0}"
SISDR_WEIGHT="${SISDR_WEIGHT:-0.1}"
MD_REGULARIZATION="${MD_REGULARIZATION:-1e-5}"
MD_COVARIANCE_TYPE="${MD_COVARIANCE_TYPE:-full}"
MD_DOMAIN_STRATEGY="${MD_DOMAIN_STRATEGY:-source_target_min}"

WANDB_PROJECT="${WANDB_PROJECT:-2026_sep_train}"
ENABLE_LOGGING="${ENABLE_LOGGING:-true}"

LOG_DIR="exp1/${EXP_NAME}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_nohup.log"

CMD=(
  python train_sep.py
  --train
  --exp "${EXP_NAME}"
  --devices "${GPU}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --extract_batch_size "${EXTRACT_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --precision "${PRECISION}"
  --train_path "${TRAIN_CSV}"
  --dev_train_path "${DEV_TRAIN_CSV}"
  --val_path "${VAL_CSV}"
  --sample_rate "${SAMPLE_RATE}"
  --segment_seconds "${SEGMENT_SECONDS}"
  --snr_min_db "${SNR_MIN_DB}"
  --snr_max_db "${SNR_MAX_DB}"
  --interference_mode "${INTERFERENCE_MODE}"
  --sep_encoder_channels "${SEP_ENCODER_CHANNELS}"
  --sep_output_mode "${SEP_OUTPUT_MODE}"
  --sep_upsample_mode "${SEP_UPSAMPLE_MODE}"
  --feature_pooling "${FEATURE_POOLING}"
  --feature_aggregation "${FEATURE_AGGREGATION}"
  --optimizer "${OPTIMIZER}"
  --scheduler "${SCHEDULER}"
  --max_lr "${MAX_LR}"
  --min_lr "${MIN_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --restart_period "${RESTART_PERIOD}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --accumulation_steps "${ACCUMULATION_STEPS}"
  --l1_spec_weight "${L1_SPEC_WEIGHT}"
  --sisdr_weight "${SISDR_WEIGHT}"
  --md_regularization "${MD_REGULARIZATION}"
  --md_covariance_type "${MD_COVARIANCE_TYPE}"
  --md_domain_strategy "${MD_DOMAIN_STRATEGY}"
)

if [[ -n "${NOISE_CSV}" ]]; then
  CMD+=(--noise_csv_path "${NOISE_CSV}")
fi

if [[ "${ENABLE_LOGGING}" == "true" ]]; then
  CMD+=(--logging --wandb_project "${WANDB_PROJECT}")
else
  CMD+=(--no-logging)
fi

echo "Launching training with nohup..."
printf 'Command: %q ' "${CMD[@]}"
printf '\n'

echo "Log file: ${LOG_FILE}"
nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "Started PID: ${PID}"
echo "Tail logs with: tail -f ${LOG_FILE}"
