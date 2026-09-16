#!/usr/bin/env bash
# BaFRC — FewRel training and evaluation script
# Usage: edit the parameters below, then run
#   bash run_fewrel.sh
# or
#   chmod +x run_fewrel.sh && ./run_fewrel.sh

set -euo pipefail

# =============================================================================
# Environment and paths
# =============================================================================
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# Data paths
# =============================================================================
ROOT="${ROOT:-./data/fewrel}"
TRAIN="train_wiki"
VAL="val_wiki"
TEST="test_wiki"
PID2NAME="description_pool"
PRETRAIN_CKPT="bert-base-uncased"

# =============================================================================
# Few-shot settings
# =============================================================================
N=5
K=1
Q=1
NA_RATE=5          # Train with 50% NOTA; evaluation covers 15%, 30%, and 50%.
SEED=5

# =============================================================================
# Training hyperparameters
# =============================================================================
BATCH_SIZE=2
TRAIN_ITER=30000
VAL_ITER=1000
TEST_ITER=10000
VAL_STEP=1000
GRAD_ITER=1
LR=2e-5
WEIGHT_DECAY=1e-5
MAX_LENGTH=128
HIDDEN_SIZE=768

# =============================================================================
# BaFRC model hyperparameters
# =============================================================================
BAFRC_MARGIN=0.15
BAFRC_CLS_TEMP=10.0
BAFRC_GAMMA=3.0
RADIUS_QUANTILE=0.1
RADIUS_REG=0.1
RADIUS_BLEND_RHO=1.0              # FewRel uses a purely query-based training radius.
BAFRC_DIST_TYPE="euclidean"          # euclidean | cosine
USE_QUERY_IN_RADIUS="true"         # true | false
USE_DESC_NEG="true"                # true | false
USE_QUERY_IN_BOUNDARY="false"      # true | false
RADIUS_UPDATE_INTERVAL=100
USE_STD_DESC="true"                # true | false
BAFRC_LOSS_POS="true"                # true | false
BAFRC_LOSS_NEG="true"                # true | false
BAFRC_LOSS_RADIUS_REG="true"         # true | false

# =============================================================================
# Run mode: set exactly one option to 1.
# =============================================================================
MODE_TRAIN=1                       # Train and evaluate when training finishes.
MODE_ONLY_TEST=0                   # Evaluate only; requires LOAD_CKPT.
MODE_TEST_ONLINE=0                 # Online submission; requires LOAD_CKPT and TEST_INPUT.

# Checkpoint for evaluation or online submission.
LOAD_CKPT=""
SAVE_CKPT=""                       # Leave empty to save under checkpoint/ automatically.

# Online submission settings
TEST_INPUT="test_wiki_input-5-1-0.15"
TEST_OUTPUT="pred-5-1.json"

# Optional encoder-only checkpoint
ENCODER_CKPT=""

# =============================================================================
# Logging; leave empty to print only to the terminal.
# =============================================================================
LOG_FILE=""                        # Example: log/fewrel-5-1-seed5.log

# =============================================================================
# Command assembly
# =============================================================================
CMD=(
  "${PYTHON}" train_fewrel.py
  --root "${ROOT}"
  --train "${TRAIN}"
  --val "${VAL}"
  --test "${TEST}"
  --pid2name "${PID2NAME}"
  --pretrain_ckpt "${PRETRAIN_CKPT}"
  --N "${N}"
  --K "${K}"
  --Q "${Q}"
  --na_rate "${NA_RATE}"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --train_iter "${TRAIN_ITER}"
  --val_iter "${VAL_ITER}"
  --test_iter "${TEST_ITER}"
  --val_step "${VAL_STEP}"
  --grad_iter "${GRAD_ITER}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_length "${MAX_LENGTH}"
  --hidden_size "${HIDDEN_SIZE}"
  --bafrc_margin "${BAFRC_MARGIN}"
  --bafrc_cls_temp "${BAFRC_CLS_TEMP}"
  --bafrc_gamma "${BAFRC_GAMMA}"
  --radius_quantile "${RADIUS_QUANTILE}"
  --radius_reg "${RADIUS_REG}"
  --radius_blend_rho "${RADIUS_BLEND_RHO}"
  --bafrc_dist_type "${BAFRC_DIST_TYPE}"
  --use_query_in_radius "${USE_QUERY_IN_RADIUS}"
  --use_desc_neg "${USE_DESC_NEG}"
  --use_query_in_boundary "${USE_QUERY_IN_BOUNDARY}"
  --radius_update_interval "${RADIUS_UPDATE_INTERVAL}"
  --use_std_desc "${USE_STD_DESC}"
  --bafrc_loss_pos "${BAFRC_LOSS_POS}"
  --bafrc_loss_neg "${BAFRC_LOSS_NEG}"
  --bafrc_loss_radius_reg "${BAFRC_LOSS_RADIUS_REG}"
)

if [[ -n "${SAVE_CKPT}" ]]; then
  CMD+=(--save_ckpt "${SAVE_CKPT}")
fi

if [[ -n "${ENCODER_CKPT}" ]]; then
  CMD+=(--encoder_ckpt "${ENCODER_CKPT}")
fi

if [[ "${MODE_TRAIN}" -eq 1 ]]; then
  :
elif [[ "${MODE_ONLY_TEST}" -eq 1 ]]; then
  if [[ -z "${LOAD_CKPT}" ]]; then
    echo "[ERROR] MODE_ONLY_TEST=1 requires LOAD_CKPT" >&2
    exit 1
  fi
  CMD+=(--only_test --load_ckpt "${LOAD_CKPT}")
elif [[ "${MODE_TEST_ONLINE}" -eq 1 ]]; then
  if [[ -z "${LOAD_CKPT}" ]]; then
    echo "[ERROR] MODE_TEST_ONLINE=1 requires LOAD_CKPT" >&2
    exit 1
  fi
  CMD+=(--only_test --test_online --load_ckpt "${LOAD_CKPT}")
  CMD+=(--test_input "${TEST_INPUT}" --test_output "${TEST_OUTPUT}")
else
  echo "[ERROR] Set exactly one of MODE_TRAIN, MODE_ONLY_TEST, or MODE_TEST_ONLINE to 1" >&2
  exit 1
fi

echo "BaFRC FewRel"

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  "${CMD[@]}"
fi
