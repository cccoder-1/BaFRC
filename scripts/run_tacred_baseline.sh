#!/usr/bin/env bash
# Reproduce BaFRC on FS-TACRED (5-way 1-shot and 5-shot).

set -euo pipefail

PYTHON="${PYTHON:-python}"
TACRED_ROOT="${TACRED_ROOT:-../TACRED_episodes}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-../bert-base-uncased}"
SHOTS=(${SHOTS:-1 5})
SEED="${SEED:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_ROOT="log/baseline/tacred"
CKPT_ROOT="checkpoint/baseline/tacred"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
TEST_SEEDS=(160290 160291 160292 160293 160294)
mkdir -p "${LOG_ROOT}" "${CKPT_ROOT}"

check_json() {
  if [[ ! -f "$1.json" ]]; then
    echo "[ERROR] Missing data file: $1.json" >&2
    exit 1
  fi
}

run_one() {
  local K="$1"
  if [[ "${K}" != "1" && "${K}" != "5" ]]; then
    echo "[ERROR] SHOTS accepts only 1 and 5." >&2
    exit 1
  fi

  local train_file="train_5w_${K}s_3q_50K_seed_123"
  local val_file="dev_5w_${K}s_3q_10K_seed_123"
  local ckpt="${CKPT_ROOT}/BaFRC-5way-${K}shot-seed${SEED}.pth.tar"
  local log_dir="${LOG_ROOT}/5way_${K}shot/seed${SEED}"
  local eval_dir="${log_dir}/eval"
  mkdir -p "${eval_dir}"

  check_json "${TACRED_ROOT}/${train_file}"
  check_json "${TACRED_ROOT}/${val_file}"
  check_json "${TACRED_ROOT}/description_pool"

  echo "[TRAIN] FS-TACRED 5-way ${K}-shot"
  "${PYTHON}" train_tacred.py \
    --root "${TACRED_ROOT}" \
    --train "${train_file}" --val "${val_file}" --test "${val_file}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --K "${K}" --seed "${SEED}" \
    --test_iter 5000 \
    --train_known_ratio 0.3 \
    --radius_max 10 \
    --save_ckpt "${ckpt}" \
    2>&1 | tee "${log_dir}/train.log"

  for test_seed in "${TEST_SEEDS[@]}"; do
    local test_file="test_episodes/5_way_${K}_shots_10K_episodes_3q_seed_${test_seed}"
    check_json "${TACRED_ROOT}/${test_file}"
    echo "[EVAL] ${test_file}"
    "${PYTHON}" train_tacred.py \
      --root "${TACRED_ROOT}" \
      --train "${train_file}" --val "${val_file}" --test "${test_file}" \
      --pretrain_ckpt "${PRETRAIN_CKPT}" \
      --K "${K}" --seed "${SEED}" \
      --test_iter 5000 \
      --radius_max 10 \
      --only_test --load_ckpt "${ckpt}" \
      2>&1 | tee "${eval_dir}/${test_seed}.log"
  done

  "${PYTHON}" - "${eval_dir}" "${SUMMARY_FILE}" "${K}" "${SEED}" <<'PY'
import re
import sys
from pathlib import Path

eval_dir, summary, shot, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
pattern = re.compile(
    r"\[FS_TACRED test\].*?ACC: ([0-9.]+), F1: ([0-9.]+).*?"
    r"target_micro P/R/F1: ([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
rows = [tuple(map(float, pattern.findall(path.read_text())[-1])) for path in sorted(eval_dir.glob("*.log"))]
if len(rows) != 5:
    raise SystemExit(f"Expected five test results, found {len(rows)}")
mean = [sum(row[i] for row in rows) / len(rows) for i in range(5)]
if not summary.exists():
    summary.write_text("K\tseed\tACC\tMacroF1\tTargetP\tTargetR\tTargetF1\n")
with summary.open("a") as handle:
    handle.write(f"{shot}\t{seed}\t" + "\t".join(f"{value:.2f}" for value in mean) + "\n")
print(f"[SUMMARY] K={shot} | target P/R/F1={mean[2]:.2f}/{mean[3]:.2f}/{mean[4]:.2f}")
PY
}

for K in "${SHOTS[@]}"; do
  run_one "${K}"
done
