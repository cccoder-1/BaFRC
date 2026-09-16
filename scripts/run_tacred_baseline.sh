#!/usr/bin/env bash
# BaFRC — FS-TACRED baseline
#
# Runs the formal BaFRC setting on FS-TACRED:
#   - 5-way 1-shot and 5-way 5-shot
#   - train/dev episodes generated with seed_123
#   - evaluation on five independent test episode seeds
#   - training-only weighted sampling increases known-query episodes
#   - validation and test retain their original episode order
#   - FS-TACRED training radius uses blended radius: rho=0.5
#
# Usage:
#   cd BaFRC
#   bash run_tacred_baseline.sh
#
# Useful overrides:
#   SEEDS="15" bash run_tacred_baseline.sh
#   SEEDS="5 10 15 20 25" bash run_tacred_baseline.sh
#   SHOTS="1" bash run_tacred_baseline.sh
#   SKIP_EXISTING=1 bash run_tacred_baseline.sh
#   EVAL_ONLY=1 bash run_tacred_baseline.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# Data
# =============================================================================
TACRED_ROOT="${TACRED_ROOT:-../TACRED_episodes}"
PID2NAME="${PID2NAME:-description_pool}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-../bert-base-uncased}"

TRAIN_1S="train_5w_1s_3q_50K_seed_123"
VAL_1S="dev_5w_1s_3q_10K_seed_123"
TRAIN_5S="train_5w_5s_3q_50K_seed_123"
VAL_5S="dev_5w_5s_3q_10K_seed_123"

TEST_SEEDS_1S=(
  "test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290"
  "test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291"
  "test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292"
  "test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293"
  "test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294"
)

TEST_SEEDS_5S=(
  "test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290"
  "test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291"
  "test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292"
  "test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293"
  "test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294"
)

# =============================================================================
# Experiment matrix
# =============================================================================
N=5
Q=1
SEEDS=(${SEEDS:-5})
SHOTS=(${SHOTS:-1 5})
SKIP_EXISTING="${SKIP_EXISTING:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

# =============================================================================
# Hyperparameters (paper setting)
# =============================================================================
BATCH_SIZE=2
TRAIN_ITER=30000
VAL_ITER=1000
# Each test file contains 10,000 episodes; batch size 2 requires 5,000 batches.
TEST_ITER=5000
VAL_STEP=1000
EARLY_STOPPING_PATIENCE=6
GRAD_ITER=1
LR=2e-5
WEIGHT_DECAY=1e-5
MAX_LENGTH=128
HIDDEN_SIZE=768

BAFRC_MARGIN=0.15
BAFRC_CLS_TEMP=10.0
BAFRC_GAMMA=3.0
RADIUS_QUANTILE="${RADIUS_QUANTILE:-0.10}"
RADIUS_BLEND_RHO=0.5
RADIUS_MAX="${RADIUS_MAX:-10}"
TRAIN_KNOWN_RATIO="${TRAIN_KNOWN_RATIO:-0.3}"
BAFRC_DIST_TYPE="euclidean"
USE_STD_DESC="true"

LOG_ROOT="log/baseline/tacred"
CKPT_ROOT="checkpoint/baseline/tacred"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"

mkdir -p "${LOG_ROOT}" "${CKPT_ROOT}"

check_file() {
  local path="$1"
  if [[ ! -f "${path}.json" ]]; then
    echo "[ERROR] Missing data file: ${path}.json" >&2
    exit 1
  fi
}

run_one() {
  local K="$1"
  local seed="$2"

  local train_file val_file
  local -a test_files
  if [[ "${K}" == "1" ]]; then
    train_file="${TRAIN_1S}"
    val_file="${VAL_1S}"
    test_files=("${TEST_SEEDS_1S[@]}")
  elif [[ "${K}" == "5" ]]; then
    train_file="${TRAIN_5S}"
    val_file="${VAL_5S}"
    test_files=("${TEST_SEEDS_5S[@]}")
  else
    echo "[ERROR] Unsupported K=${K}. Use 1 or 5." >&2
    exit 1
  fi

  check_file "${TACRED_ROOT}/${train_file}"
  check_file "${TACRED_ROOT}/${val_file}"
  check_file "${TACRED_ROOT}/${PID2NAME}.json"
  for test_file in "${test_files[@]}"; do
    check_file "${TACRED_ROOT}/${test_file}"
  done

  local tag="BaFRC-baseline-5way-${K}shot-seed${seed}"
  local log_dir="${LOG_ROOT}/5way_${K}shot/seed${seed}"
  local train_log="${log_dir}/train.log"
  local eval_dir="${log_dir}/eval"
  local ckpt="${CKPT_ROOT}/${tag}.pth.tar"

  mkdir -p "${log_dir}" "${eval_dir}"

  echo ""
  echo "============================================================"
  echo "BaFRC FS-TACRED baseline"
  echo "Setting : 5-way ${K}-shot | seed=${seed}"
  echo "Train   : ${train_file}"
  echo "Val     : ${val_file}"
  echo "Train known ratio: ${TRAIN_KNOWN_RATIO}"
  echo "Validation/test sampling: original order"
  echo "Ckpt    : ${ckpt}"
  echo "Log     : ${train_log}"
  echo "============================================================"

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
      echo "[SKIP train] checkpoint exists: ${ckpt}"
    else
      "${PYTHON}" train_tacred.py \
        --model BaFRC \
        --root "${TACRED_ROOT}" \
        --train "${train_file}" --val "${val_file}" --test "${val_file}" \
        --pid2name "${PID2NAME}" --pretrain_ckpt "${PRETRAIN_CKPT}" \
        --N "${N}" --K "${K}" --Q "${Q}" --seed "${seed}" \
        --batch_size "${BATCH_SIZE}" \
        --train_iter "${TRAIN_ITER}" --val_iter "${VAL_ITER}" \
        --test_iter "${TEST_ITER}" --val_step "${VAL_STEP}" \
        --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
        --grad_iter "${GRAD_ITER}" --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" \
        --max_length "${MAX_LENGTH}" --hidden_size "${HIDDEN_SIZE}" \
        --bafrc_margin "${BAFRC_MARGIN}" \
        --bafrc_cls_temp "${BAFRC_CLS_TEMP}" \
        --bafrc_gamma "${BAFRC_GAMMA}" \
        --radius_quantile "${RADIUS_QUANTILE}" \
        --radius_blend_rho "${RADIUS_BLEND_RHO}" \
        --radius_max "${RADIUS_MAX}" \
        --train_known_ratio "${TRAIN_KNOWN_RATIO}" \
        --bafrc_dist_type "${BAFRC_DIST_TYPE}" \
        --use_std_desc "${USE_STD_DESC}" \
        --save_ckpt "${ckpt}" \
        2>&1 | tee "${train_log}"
    fi
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] checkpoint not found for evaluation: ${ckpt}" >&2
    exit 1
  fi

  for test_file in "${test_files[@]}"; do
    local tb eval_log
    tb="$(basename "${test_file}")"
    eval_log="${eval_dir}/${tb}.log"
    echo ""
    echo "[EVAL] ${test_file}"
    "${PYTHON}" train_tacred.py \
      --model BaFRC \
      --root "${TACRED_ROOT}" \
      --train "${train_file}" --val "${val_file}" --test "${test_file}" \
      --pid2name "${PID2NAME}" --pretrain_ckpt "${PRETRAIN_CKPT}" \
      --N "${N}" --K "${K}" --Q "${Q}" --seed "${seed}" \
      --batch_size "${BATCH_SIZE}" --test_iter "${TEST_ITER}" \
      --max_length "${MAX_LENGTH}" --hidden_size "${HIDDEN_SIZE}" \
      --bafrc_margin "${BAFRC_MARGIN}" \
      --bafrc_cls_temp "${BAFRC_CLS_TEMP}" \
      --bafrc_gamma "${BAFRC_GAMMA}" \
      --radius_quantile "${RADIUS_QUANTILE}" \
      --radius_blend_rho "${RADIUS_BLEND_RHO}" \
      --radius_max "${RADIUS_MAX}" \
      --bafrc_dist_type "${BAFRC_DIST_TYPE}" \
      --use_std_desc "${USE_STD_DESC}" \
      --only_test --load_ckpt "${ckpt}" \
      2>&1 | tee "${eval_log}"
  done

  "${PYTHON}" - <<PY
import re
from pathlib import Path

summary = Path("${SUMMARY_FILE}")
eval_dir = Path("${eval_dir}")
rows = []
pattern = re.compile(
    r"\\[FS_TACRED test\\].*?ACC: ([0-9.]+), F1: ([0-9.]+).*?"
    r"target_micro P/R/F1: ([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
for log in sorted(eval_dir.glob("*.log")):
    text = log.read_text()
    matches = pattern.findall(text)
    if not matches:
        rows.append((log.name, None))
        continue
    acc, macro_f1, p, r, f1 = map(float, matches[-1])
    rows.append((log.name, (acc, macro_f1, p, r, f1)))

valid = [x for _, x in rows if x is not None]
if not valid:
    raise SystemExit("No parsable eval results found in " + str(eval_dir))

mean = [sum(v[i] for v in valid) / len(valid) for i in range(5)]
all_target_f1 = ",".join(f"{v[4]:.2f}" for v in valid)

if not summary.exists():
    summary.write_text(
        "model\\tK\\tseed\\ttrain_known_ratio\\tradius_max\\tACC\\tMacroF1\\t"
        "TargetP\\tTargetR\\tTargetF1\\tAllTargetF1\\tlog_dir\\n"
    )
with summary.open("a") as f:
    f.write(
        f"BaFRC\\t${K}\\t${seed}\\t${TRAIN_KNOWN_RATIO}\\t${RADIUS_MAX}\\t"
        f"{mean[0]:.2f}\\t{mean[1]:.2f}\\t{mean[2]:.2f}\\t{mean[3]:.2f}\\t{mean[4]:.2f}\\t"
        f"{all_target_f1}\\t{eval_dir}\\n"
    )

print(
    f"[SUMMARY] K=${K} seed=${seed} | target P/R/F1 = "
    f"{mean[2]:.2f}/{mean[3]:.2f}/{mean[4]:.2f}"
)
PY
}

echo "============================================================"
echo "BaFRC FS-TACRED baseline"
echo "TACRED_ROOT=${TACRED_ROOT}"
echo "Shots=${SHOTS[*]} | Seeds=${SEEDS[*]}"
echo "TRAIN_KNOWN_RATIO=${TRAIN_KNOWN_RATIO} | RADIUS_MAX=${RADIUS_MAX}"
echo "Summary=${SUMMARY_FILE}"
echo "============================================================"

for K in "${SHOTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${K}" "${seed}"
  done
done

echo ""
echo "All done. Summary: ${SUMMARY_FILE}"
