#!/usr/bin/env bash
# Reproduce the BaFRC FewRel result. Override paths or the shot with environment variables.

set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="${ROOT:-./data/fewrel}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-bert-base-uncased}"
K="${K:-1}"
SEED="${SEED:-5}"
LOAD_CKPT="${LOAD_CKPT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

CMD=(
  "${PYTHON}" train_fewrel.py
  --root "${ROOT}"
  --pretrain_ckpt "${PRETRAIN_CKPT}"
  --K "${K}"
  --seed "${SEED}"
  --na_rate 5
  --radius_blend_rho 1.0
)

if [[ -n "${LOAD_CKPT}" ]]; then
  CMD+=(--only_test --load_ckpt "${LOAD_CKPT}")
fi

"${CMD[@]}"
