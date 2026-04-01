#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Source-separation proxy training launcher (nohup-friendly)
# Extended for:
#   - external separator pretraining (e.g. AudioSep-style checkpoint)
#   - optional stage2 SED guide branch
#   - Time-FiLM / latent injection / DPRNN / iterative refinement
# -----------------------------------------------------------------------------
# Usage examples:
#   bash scripts/train_sep_nohup.sh
#   EXP_NAME=sep_resunet_audiosep GPU=1 \
#   PRETRAINED_SEP_CKPT=/path/to/audiosep.ckpt \
#   bash scripts/train_sep_nohup.sh
#
# If you have not yet renamed train_sep_modified.py back to train_sep.py,
# set TRAIN_SCRIPT explicitly:
#   TRAIN_SCRIPT=train_sep_modified.py bash scripts/train_sep_nohup.sh
# -----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_sep.py}"

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
TRAIN_FIXED_SNR_DB="${TRAIN_FIXED_SNR_DB:-}"
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

# -----------------------------------------------------------------------------
# External pretrained initialization (actual transfer learning)
# -----------------------------------------------------------------------------
PRETRAINED_SEP_CKPT="${PRETRAINED_SEP_CKPT:-}"
PRETRAINED_SEP_STRICT_BACKBONE="${PRETRAINED_SEP_STRICT_BACKBONE:-false}"
PRETRAINED_GUIDE_CKPT="${PRETRAINED_GUIDE_CKPT:-}"
PRETRAINED_GUIDE_STRICT="${PRETRAINED_GUIDE_STRICT:-false}"
LOAD_EXTERNAL_PRETRAINED_IN_DRIVER="${LOAD_EXTERNAL_PRETRAINED_IN_DRIVER:-true}"

# -----------------------------------------------------------------------------
# Guide conditioning / 2nd-place-team style extensions
# -----------------------------------------------------------------------------
GUIDE_CLASS_MODE="${GUIDE_CLASS_MODE:-machine}"
GUIDE_RETURN_REFERENCE_WAVE="${GUIDE_RETURN_REFERENCE_WAVE:-false}"
GUIDE_REFERENCE_STRATEGY="${GUIDE_REFERENCE_STRATEGY:-same_class_random}"

USE_STAGE2_SED="${USE_STAGE2_SED:-false}"
GUIDE_INPUT_CHANNELS="${GUIDE_INPUT_CHANNELS:-1}"
GUIDE_NUM_CLASSES="${GUIDE_NUM_CLASSES:-1}"
GUIDE_DEFAULT_CLASS_INDEX="${GUIDE_DEFAULT_CLASS_INDEX:-0}"
GUIDE_STEM_CHANNELS="${GUIDE_STEM_CHANNELS:-64}"
GUIDE_HIDDEN_DIM="${GUIDE_HIDDEN_DIM:-256}"
GUIDE_NUM_LAYERS="${GUIDE_NUM_LAYERS:-4}"
GUIDE_NUM_HEADS="${GUIDE_NUM_HEADS:-4}"
GUIDE_MLP_RATIO="${GUIDE_MLP_RATIO:-4.0}"
GUIDE_DROPOUT="${GUIDE_DROPOUT:-0.1}"
GUIDE_TEMPORAL_CONV_KERNEL_SIZE="${GUIDE_TEMPORAL_CONV_KERNEL_SIZE:-3}"
GUIDE_MAX_TIME_POSITIONS="${GUIDE_MAX_TIME_POSITIONS:-2048}"
GUIDE_USE_FREQUENCY_ATTENTION_POOL="${GUIDE_USE_FREQUENCY_ATTENTION_POOL:-true}"
GUIDE_RETURN_ALL_HIDDEN_STATES="${GUIDE_RETURN_ALL_HIDDEN_STATES:-true}"
GUIDE_STRONG_ACTIVATION="${GUIDE_STRONG_ACTIVATION:-sigmoid}"

USE_TIME_FILM="${USE_TIME_FILM:-false}"
TIME_FILM_CONDITION_DIM="${TIME_FILM_CONDITION_DIM:-1}"
TIME_FILM_HIDDEN_DIM="${TIME_FILM_HIDDEN_DIM:-128}"
TIME_FILM_NUM_LAYERS="${TIME_FILM_NUM_LAYERS:-2}"
TIME_FILM_DROPOUT="${TIME_FILM_DROPOUT:-0.0}"
TIME_FILM_ON_BOTTLENECK="${TIME_FILM_ON_BOTTLENECK:-true}"
TIME_FILM_ON_DECODER="${TIME_FILM_ON_DECODER:-false}"
TIME_FILM_RESIDUAL_GAMMA="${TIME_FILM_RESIDUAL_GAMMA:-true}"

USE_LATENT_INJECTION="${USE_LATENT_INJECTION:-false}"
LATENT_INJECTION_INPUT_DIM="${LATENT_INJECTION_INPUT_DIM:-}"
LATENT_INJECTION_HIDDEN_DIM="${LATENT_INJECTION_HIDDEN_DIM:-128}"
LATENT_INJECTION_NUM_HIDDEN_STATES="${LATENT_INJECTION_NUM_HIDDEN_STATES:-}"

USE_DPRNN="${USE_DPRNN:-false}"
DPRNN_HIDDEN_SIZE="${DPRNN_HIDDEN_SIZE:-256}"
DPRNN_NUM_LAYERS="${DPRNN_NUM_LAYERS:-1}"
DPRNN_DROPOUT="${DPRNN_DROPOUT:-0.0}"
DPRNN_RNN_TYPE="${DPRNN_RNN_TYPE:-gru}"
DPRNN_BIDIRECTIONAL="${DPRNN_BIDIRECTIONAL:-true}"

USE_ITERATIVE_REFINEMENT="${USE_ITERATIVE_REFINEMENT:-false}"
REFINEMENT_NUM_ITERATIONS="${REFINEMENT_NUM_ITERATIONS:-2}"
REFINEMENT_CHANNELS="${REFINEMENT_CHANNELS:-1}"
REFINEMENT_ADAPTER_HIDDEN_CHANNELS="${REFINEMENT_ADAPTER_HIDDEN_CHANNELS:-}"
REFINEMENT_ADAPTER_NUM_LAYERS="${REFINEMENT_ADAPTER_NUM_LAYERS:-1}"
REFINEMENT_ADAPTER_ACTIVATION="${REFINEMENT_ADAPTER_ACTIVATION:-relu}"
REFINEMENT_DETACH_BETWEEN_ITERATIONS="${REFINEMENT_DETACH_BETWEEN_ITERATIONS:-true}"
REFINEMENT_RESIDUAL_TO_BASE_INPUT="${REFINEMENT_RESIDUAL_TO_BASE_INPUT:-false}"
REFINEMENT_RETURN_HISTORY="${REFINEMENT_RETURN_HISTORY:-false}"
REFINEMENT_SIGNAL_KEY="${REFINEMENT_SIGNAL_KEY:-pred_spec}"

LOG_DIR="exp1/${EXP_NAME}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_nohup.log"

CMD=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
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
  --guide_class_mode "${GUIDE_CLASS_MODE}"
  --guide_reference_strategy "${GUIDE_REFERENCE_STRATEGY}"
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
  --guide_input_channels "${GUIDE_INPUT_CHANNELS}"
  --guide_num_classes "${GUIDE_NUM_CLASSES}"
  --guide_default_class_index "${GUIDE_DEFAULT_CLASS_INDEX}"
  --guide_stem_channels "${GUIDE_STEM_CHANNELS}"
  --guide_hidden_dim "${GUIDE_HIDDEN_DIM}"
  --guide_num_layers "${GUIDE_NUM_LAYERS}"
  --guide_num_heads "${GUIDE_NUM_HEADS}"
  --guide_mlp_ratio "${GUIDE_MLP_RATIO}"
  --guide_dropout "${GUIDE_DROPOUT}"
  --guide_temporal_conv_kernel_size "${GUIDE_TEMPORAL_CONV_KERNEL_SIZE}"
  --guide_max_time_positions "${GUIDE_MAX_TIME_POSITIONS}"
  --guide_strong_activation "${GUIDE_STRONG_ACTIVATION}"
  --time_film_condition_dim "${TIME_FILM_CONDITION_DIM}"
  --time_film_hidden_dim "${TIME_FILM_HIDDEN_DIM}"
  --time_film_num_layers "${TIME_FILM_NUM_LAYERS}"
  --time_film_dropout "${TIME_FILM_DROPOUT}"
  --latent_injection_hidden_dim "${LATENT_INJECTION_HIDDEN_DIM}"
  --dprnn_hidden_size "${DPRNN_HIDDEN_SIZE}"
  --dprnn_num_layers "${DPRNN_NUM_LAYERS}"
  --dprnn_dropout "${DPRNN_DROPOUT}"
  --dprnn_rnn_type "${DPRNN_RNN_TYPE}"
  --refinement_num_iterations "${REFINEMENT_NUM_ITERATIONS}"
  --refinement_channels "${REFINEMENT_CHANNELS}"
  --refinement_adapter_num_layers "${REFINEMENT_ADAPTER_NUM_LAYERS}"
  --refinement_adapter_activation "${REFINEMENT_ADAPTER_ACTIVATION}"
  --refinement_signal_key "${REFINEMENT_SIGNAL_KEY}"
)

if [[ -n "${NOISE_CSV}" ]]; then
  CMD+=(--noise_csv_path "${NOISE_CSV}")
fi
if [[ -n "${TRAIN_FIXED_SNR_DB}" ]]; then
  CMD+=(--train_fixed_snr_db "${TRAIN_FIXED_SNR_DB}")
fi

# External pretrained initialization
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

# Optional guide/reference controls
if [[ "${GUIDE_RETURN_REFERENCE_WAVE}" == "true" ]]; then
  CMD+=(--guide_return_reference_wave)
fi
if [[ "${GUIDE_USE_FREQUENCY_ATTENTION_POOL}" == "false" ]]; then
  CMD+=(--no-guide_use_frequency_attention_pool)
fi
if [[ "${GUIDE_RETURN_ALL_HIDDEN_STATES}" == "false" ]]; then
  CMD+=(--no-guide_return_all_hidden_states)
fi

# 2nd-place-team style feature switches
if [[ "${USE_STAGE2_SED}" == "true" ]]; then
  CMD+=(--use_stage2_sed)
fi
if [[ "${USE_TIME_FILM}" == "true" ]]; then
  CMD+=(--use_time_film)
fi
if [[ "${TIME_FILM_ON_BOTTLENECK}" == "false" ]]; then
  CMD+=(--no-time_film_on_bottleneck)
fi
if [[ "${TIME_FILM_ON_DECODER}" == "true" ]]; then
  CMD+=(--time_film_on_decoder)
fi
if [[ "${TIME_FILM_RESIDUAL_GAMMA}" == "false" ]]; then
  CMD+=(--no-time_film_residual_gamma)
fi
if [[ "${USE_LATENT_INJECTION}" == "true" ]]; then
  CMD+=(--use_latent_injection)
fi
if [[ -n "${LATENT_INJECTION_INPUT_DIM}" ]]; then
  CMD+=(--latent_injection_input_dim "${LATENT_INJECTION_INPUT_DIM}")
fi
if [[ -n "${LATENT_INJECTION_NUM_HIDDEN_STATES}" ]]; then
  CMD+=(--latent_injection_num_hidden_states "${LATENT_INJECTION_NUM_HIDDEN_STATES}")
fi
if [[ "${USE_DPRNN}" == "true" ]]; then
  CMD+=(--use_dprnn)
fi
if [[ "${DPRNN_BIDIRECTIONAL}" == "false" ]]; then
  CMD+=(--no-dprnn_bidirectional)
fi
if [[ "${USE_ITERATIVE_REFINEMENT}" == "true" ]]; then
  CMD+=(--use_iterative_refinement)
fi
if [[ -n "${REFINEMENT_ADAPTER_HIDDEN_CHANNELS}" ]]; then
  CMD+=(--refinement_adapter_hidden_channels "${REFINEMENT_ADAPTER_HIDDEN_CHANNELS}")
fi
if [[ "${REFINEMENT_DETACH_BETWEEN_ITERATIONS}" == "false" ]]; then
  CMD+=(--no-refinement_detach_between_iterations)
fi
if [[ "${REFINEMENT_RESIDUAL_TO_BASE_INPUT}" == "true" ]]; then
  CMD+=(--refinement_residual_to_base_input)
fi
if [[ "${REFINEMENT_RETURN_HISTORY}" == "true" ]]; then
  CMD+=(--refinement_return_history)
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
